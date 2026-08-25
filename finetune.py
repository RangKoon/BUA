# finetune.py
# Qwen2.5-32B LoRA 파인튜닝 스크립트
# B200 GPU 서버 (183GB VRAM) 기준
# 충분한 VRAM으로 4bit 양자화 불필요 → 일반 LoRA 사용

import json
import os
from pathlib import Path
from dataclasses import dataclass, field

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
)


# ──────────────────────────────────────────
# 설정
# ──────────────────────────────────────────
@dataclass
class Config:
    # 모델
    model_name: str = "Qwen/Qwen2.5-32B-Instruct"

    # 데이터
    data_path: str = "./data/homepage/ft_homepage_success.jsonl"

    # 출력
    output_dir: str = "./output/qwen2.5-32b-cnu-lora"

    # LoRA 설정 (B200이라 rank 크게)
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    # 학습 설정
    num_epochs: int = 3
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 16   # effective batch = 16
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.1
    max_seq_length: int = 16384             # batch=1로 OOM 방지하면서 전체 커버

    # 저장
    save_steps: int = 50
    logging_steps: int = 10
    eval_ratio: float = 0.05


# ──────────────────────────────────────────
# 이미지 블록 제거 (use_vision=True 데이터 대응)
# ──────────────────────────────────────────
def strip_images(messages: list[dict]) -> list[dict]:
    result = []
    for m in messages:
        content = m["content"]
        if isinstance(content, list):
            # 텍스트 블록만 추출
            text_parts = [
                c.get("text", "") for c in content
                if isinstance(c, dict) and c.get("type") == "text"
            ]
            text = " ".join(text_parts).strip()
            if text:
                result.append({"role": m["role"], "content": text})
        elif isinstance(content, str):
            result.append(m)
    return result


# ──────────────────────────────────────────
# 데이터 로드
# ──────────────────────────────────────────
def load_data(data_path: str) -> list[dict]:
    data = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                d["messages"] = strip_images(d["messages"])
                if d["messages"]:
                    data.append(d)
    print(f"총 {len(data)}개 데이터 로드 완료")
    return data


# ──────────────────────────────────────────
# 전처리: assistant 응답만 학습 대상으로
# ──────────────────────────────────────────
def preprocess_function(examples, tokenizer, max_length: int):
    input_ids_list = []
    labels_list = []
    attention_mask_list = []

    for messages in examples["messages"]:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        tokenized = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_tensors=None,
        )

        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]
        labels = input_ids.copy()

        # system/user 부분 -100 마스킹, assistant만 학습
        assistant_token = tokenizer.encode(
            "<|im_start|>assistant", add_special_tokens=False
        )
        end_token = tokenizer.encode(
            "<|im_end|>", add_special_tokens=False
        )

        mask = True
        i = 0
        while i < len(labels):
            if labels[i:i+len(assistant_token)] == assistant_token:
                mask = False
                i += len(assistant_token)
                continue
            if labels[i:i+len(end_token)] == end_token:
                if not mask:
                    labels[i] = end_token[0]
                mask = True
                i += len(end_token)
                continue
            if mask:
                labels[i] = -100
            i += 1

        input_ids_list.append(input_ids)
        labels_list.append(labels)
        attention_mask_list.append(attention_mask)

    return {
        "input_ids": input_ids_list,
        "labels": labels_list,
        "attention_mask": attention_mask_list,
    }


# ──────────────────────────────────────────
# 메인 학습
# ──────────────────────────────────────────
def train(cfg: Config):
    print(f"\n{'='*50}")
    print(f"모델: {cfg.model_name}")
    print(f"데이터: {cfg.data_path}")
    print(f"출력: {cfg.output_dir}")
    print(f"LoRA rank: {cfg.lora_r}, alpha: {cfg.lora_alpha}")
    print(f"max_seq_length: {cfg.max_seq_length}")
    print('='*50)

    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    # ── 1. 토크나이저 ──
    print("\n📦 토크나이저 로드 중…")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── 2. 데이터 ──
    print("\n📂 데이터 로드 중…")
    raw_data = load_data(cfg.data_path)

    eval_size = max(1, int(len(raw_data) * cfg.eval_ratio))
    train_data = raw_data[eval_size:]
    eval_data = raw_data[:eval_size]
    print(f"Train: {len(train_data)}개 / Eval: {len(eval_data)}개")

    train_dataset = Dataset.from_list([{"messages": d["messages"]} for d in train_data])
    eval_dataset = Dataset.from_list([{"messages": d["messages"]} for d in eval_data])

    print("\n🔧 토크나이징 중…")
    train_dataset = train_dataset.map(
        lambda x: preprocess_function(x, tokenizer, cfg.max_seq_length),
        batched=True,
        remove_columns=["messages"],
        desc="Train",
    )
    eval_dataset = eval_dataset.map(
        lambda x: preprocess_function(x, tokenizer, cfg.max_seq_length),
        batched=True,
        remove_columns=["messages"],
        desc="Eval",
    )

    # ── 3. 모델 로드 (bfloat16, 양자화 없음) ──
    print("\n🤖 모델 로드 중 (bfloat16 LoRA)…")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.enable_input_require_grads()

    # ── 4. LoRA 적용 ──
    print("\n🔩 LoRA 적용 중…")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.lora_target_modules,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── 5. 학습 설정 ──
    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type="cosine",
        bf16=True,
        fp16=False,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        eval_steps=cfg.save_steps,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        save_total_limit=3,           # 체크포인트 최대 3개 유지 (디스크 절약)
        report_to="none",
        dataloader_num_workers=4,
        remove_unused_columns=False,
        gradient_checkpointing=True,
        group_by_length=True,         # 비슷한 길이끼리 배치 → 패딩 낭비 감소
        optim="adamw_torch_fused",    # B200에 최적화
    )

    # ── 6. 학습 ──
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    print("\n🚀 학습 시작!")
    trainer.train()

    # ── 7. 저장 ──
    print("\n💾 모델 저장 중...")
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    print(f"✅ 저장 완료: {cfg.output_dir}")


if __name__ == "__main__":
    cfg = Config()
    train(cfg)
