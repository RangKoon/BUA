from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

BASE_MODEL = "Qwen/Qwen2.5-32B-Instruct"
LORA_PATH = "./output/qwen2.5-32b-cnu-lora"
MERGED_PATH = "./output/qwen2.5-32b-cnu-merged"

print("베이스 모델 로드 중...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="cpu",  # merge는 CPU에서
    trust_remote_code=True,
)

print("LoRA 가중치 로드 및 merge 중...")
model = PeftModel.from_pretrained(model, LORA_PATH)
model = model.merge_and_unload()

print("저장 중...")
model.save_pretrained(MERGED_PATH)

tokenizer = AutoTokenizer.from_pretrained(LORA_PATH)
tokenizer.save_pretrained(MERGED_PATH)

print(f"완료! → {MERGED_PATH}")
