# compare.py
# 베이스 모델 vs 파인튜닝 모델 응답 비교
# 모델을 한 번만 로드하고 LoRA를 껐다 켜며 비교

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-32B-Instruct"
LORA_PATH = "./output/qwen2.5-32b-cnu-qlora"

MAX_NEW_TOKENS = 1024
TEMPERATURE = 0.7


def load_model():
    print("토크나이저 로드 중...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

    print("베이스 모델 로드 중 (4bit)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def load_lora(model):
    print("LoRA 어댑터 로드 중...")
    model = PeftModel.from_pretrained(model, LORA_PATH)
    model.eval()
    return model


def generate(model, tokenizer, messages: list[dict]) -> str:
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    # 입력 부분 제거, 새로 생성된 토큰만 디코딩
    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def compare(model, tokenizer, system_prompt: str, user_query: str):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query},
    ]

    print("\n" + "=" * 60)
    print("[ 베이스 모델 응답 ]")
    print("=" * 60)
    if isinstance(model, PeftModel):
        with model.disable_adapter():
            base_response = generate(model, tokenizer, messages)
    else:
        base_response = generate(model, tokenizer, messages)
    print(base_response)

    print("\n" + "=" * 60)
    print("[ 파인튜닝 모델 응답 ]")
    print("=" * 60)
    if isinstance(model, PeftModel):
        ft_response = generate(model, tokenizer, messages)
    else:
        print("LoRA가 로드되지 않았습니다. LORA_PATH를 확인하세요.")
        ft_response = ""
    print(ft_response)

    return base_response, ft_response


def main():
    model, tokenizer = load_model()
    model = load_lora(model)

    # 시스템 프롬프트 - 학습 데이터와 동일하게
    system_prompt = (
        "You are an AI agent designed to operate in an iterative loop to automate browser tasks. "
        "Your ultimate goal is accomplishing the task provided in <user_request>."
    )

    print("\n모델 로드 완료! 쿼리를 입력하세요. (종료: 'exit')")

    while True:
        print("\n" + "-" * 60)
        user_input = input("쿼리 입력: ").strip()
        if user_input.lower() in ("exit", "quit", "q"):
            break
        if not user_input:
            continue

        compare(model, tokenizer, system_prompt, user_input)


if __name__ == "__main__":
    main()
