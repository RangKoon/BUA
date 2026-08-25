# finetune_vl.py
# Qwen2.5-VL-32B LoRA 파인튜닝 스크립트
# B200 GPU 서버 (183GB VRAM) 기준
# ft_homepage_success.jsonl (image_url 포함) 사용

import json
from pathlib import Path
from dataclasses import dataclass, field
from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

import torch
from torch.utils.data import Dataset as TorchDataset
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    TrainingArguments,
    Trainer,
)
from qwen_vl_utils import process_vision_info


# ──────────────────────────────────────────
# 설정
# ──────────────────────────────────────────
@dataclass
class Config:
    # 모델 (서버 내 캐시 경로)
    model_name: str = (
        "/NHNHOME/WORKSPACE/26moe002_H/mwchoi/hf_cache/hub/"
        "models--Qwen--Qwen2.5-VL-32B-Instruct/snapshots/"
        "7cfb30d71a1f4f49a57592323337a4a4727301da"
    )

    # 데이터
    data_path: str = "./data/homepage/ft_homepage_success.jsonl"

    # 출력
    output_dir: str = "./output/qwen2.5-vl-32b-cnu-lora"

    # LoRA 설정
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    # 학습 설정
    num_epochs: int = 3
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 64
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.1
    max_seq_length: int = 12288 #16384에서 줄인 상태

    use_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    use_double_quant: bool = True

    # 저장
    save_steps: int = 50
    logging_steps: int = 10
    eval_ratio: float = 0.05


# ──────────────────────────────────────────
# image_url → Qwen2.5-VL image 포맷 변환
# ──────────────────────────────────────────
def convert_image_format(messages: list[dict]) -> list[dict]:
    """
    browser-use 수집 포맷 (image_url + base64) →
    Qwen2.5-VL 포맷 (type=image, image=data:...) 변환
    """
    result = []
    for msg in messages:
        content = msg["content"]
        if isinstance(content, list):
            new_content = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "image_url":
                    url = block.get("image_url", {}).get("url", "")
                    new_content.append({"type": "image", "image": url})
                else:
                    new_content.append(block)
            result.append({"role": msg["role"], "content": new_content})
        else:
            result.append(msg)
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
                d["messages"] = convert_image_format(d["messages"])
                if d["messages"]:
                    data.append(d)
    print(f"총 {len(data)}개 데이터 로드 완료")
    return data


# ──────────────────────────────────────────
# PyTorch Dataset (이미지 처리 포함)
# ──────────────────────────────────────────
class VLDataset(TorchDataset):
    def __init__(self, data: list[dict], processor, max_length: int):
        self.processor = processor
        self.max_length = max_length
    
        self.assistant_token_ids = processor.tokenizer.encode(
            "<|im_start|>assistant", add_special_tokens=False
        )
        self.end_token_ids = processor.tokenizer.encode(
            "<|im_end|>", add_special_tokens=False
        )
    
        print("시퀀스 길이 필터링 중...")
        valid_data = []
        skipped = 0
        for item in data:
            messages = item["messages"]
            try:
                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = processor(
                    text=[text],
                    images=image_inputs if image_inputs else None,
                    videos=video_inputs if video_inputs else None,
                    padding=False,
                    return_tensors="pt",
                )
                if inputs["input_ids"].shape[1] <= max_length:
                    valid_data.append(item)
                else:
                    skipped += 1
            except Exception:
                skipped += 1
    
        self.data = valid_data
        print(f"유효 샘플: {len(self.data)}, 스킵: {skipped}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        messages = self.data[idx]["messages"]

        # 텍스트 템플릿 생성
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        # 이미지 추출
        image_inputs, video_inputs = process_vision_info(messages)

        # 토크나이징 + 이미지 인코딩
        inputs = self.processor(
            text=[text],
            images=image_inputs if image_inputs else None,
            videos=video_inputs if video_inputs else None,
            padding=False,
            return_tensors="pt",
        )

        input_ids = inputs["input_ids"][0]
        attention_mask = inputs["attention_mask"][0]

        # truncation
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            attention_mask = attention_mask[:self.max_length]

        # assistant 응답만 학습 대상으로 (나머지 -100 마스킹)
        labels = input_ids.clone().tolist()
        masking = True
        i = 0
        a_ids = self.assistant_token_ids
        e_ids = self.end_token_ids
        while i < len(labels):
            if labels[i:i+len(a_ids)] == a_ids:
                masking = False
                i += len(a_ids)
                continue
            if labels[i:i+len(e_ids)] == e_ids:
                if not masking:
                    labels[i] = e_ids[0]
                masking = True
                i += len(e_ids)
                continue
            if masking:
                labels[i] = -100
            i += 1
        labels = torch.tensor(labels)

        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        if "pixel_values" in inputs:
            result["pixel_values"] = inputs["pixel_values"]   # [total_patches, C, H, W]
        if "image_grid_thw" in inputs:
            result["image_grid_thw"] = inputs["image_grid_thw"]  # [num_images, 3]

        return result


# ──────────────────────────────────────────
# Data Collator (가변 이미지 크기 처리)
# ──────────────────────────────────────────
class VLDataCollator:
    def __init__(self, processor):
        self.pad_token_id = processor.tokenizer.pad_token_id or 0

    def __call__(self, batch):
        max_len = max(item["input_ids"].shape[0] for item in batch)

        input_ids, attention_masks, labels = [], [], []
        for item in batch:
            pad_len = max_len - item["input_ids"].shape[0]
            input_ids.append(torch.cat([
                item["input_ids"],
                torch.full((pad_len,), self.pad_token_id, dtype=torch.long)
            ]))
            attention_masks.append(torch.cat([
                item["attention_mask"],
                torch.zeros(pad_len, dtype=torch.long)
            ]))
            labels.append(torch.cat([
                item["labels"],
                torch.full((pad_len,), -100, dtype=torch.long)
            ]))

        result = {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_masks),
            "labels": torch.stack(labels),
        }

        # 이미지 텐서 concat (각 샘플의 이미지 patch 수가 다를 수 있음)
        if "pixel_values" in batch[0]:
            result["pixel_values"] = torch.cat(
                [item["pixel_values"] for item in batch], dim=0
            )
        if "image_grid_thw" in batch[0]:
            result["image_grid_thw"] = torch.cat(
                [item["image_grid_thw"] for item in batch], dim=0
            )

        return result


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

    # ── 1. Processor 로드 ──
    print("\n📦 Processor 로드 중…")
    processor = AutoProcessor.from_pretrained(
        cfg.model_name,
        trust_remote_code=True,
        padding_side="right",
        min_pixels=256*28*28,
        max_pixels=256*28*28,
    )

    # ── 2. 데이터 로드 & 분할 ──
    print("\n📂 데이터 로드 중…")
    raw_data = load_data(cfg.data_path)

    eval_size = max(1, int(len(raw_data) * cfg.eval_ratio))
    train_data = raw_data[eval_size:]
    eval_data = raw_data[:eval_size]
    print(f"Train: {len(train_data)}개 / Eval: {len(eval_data)}개")

    train_dataset = VLDataset(train_data, processor, cfg.max_seq_length)
    eval_dataset  = VLDataset(eval_data,  processor, cfg.max_seq_length)

    # ── 3. 모델 로드 (bfloat16) ──
    print("\n🤖 모델 로드 중 (bfloat16)…")

    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        cfg.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",   # 추가
    )
    model = prepare_model_for_kbit_training(model)
    model.enable_input_require_grads()
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
        save_total_limit=3,
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=True,
        # group_by_length=False,   # VL은 이미지 크기 달라 group_by_length 비활성화
        optim="paged_adamw_32bit",
        dataloader_num_workers=0,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    # ── 6. 학습 ──
    data_collator = VLDataCollator(processor)

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
    processor.save_pretrained(cfg.output_dir)
    print(f"✅ 저장 완료: {cfg.output_dir}")


if __name__ == "__main__":
    cfg = Config()
    train(cfg)
