import os
import random
import subprocess
from pathlib import Path
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier

# ---------------- 1. 参数 ----------------
MODEL_ID = "Qwen2.5-7B-Math"
LOCAL_MODEL_DIR = Path("./models") / MODEL_ID.split("/")[-1]
SAVE_QUANT_DIR = LOCAL_MODEL_DIR.parent / f"{LOCAL_MODEL_DIR.name}-W4A16-commonsense"

NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 2048

# ---------------- 2. 设置镜像（可选） ----------------
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# ---------------- 3. 加载模型 ----------------
print(">>> 从本地加载模型 …")
model = AutoModelForCausalLM.from_pretrained(
    LOCAL_MODEL_DIR,
    dtype="auto",
    device_map="auto",
    local_files_only=True
)
tokenizer = AutoTokenizer.from_pretrained(
    LOCAL_MODEL_DIR,
    local_files_only=True,
    use_fast=True
)

# ---------------- 4. 加载校准数据 ----------------
print(">>> 加载 CommonSense QA 校准数据集 …")
custom_cache_dir = "./datasets/commonsense_qa"
ds_dict = load_dataset(
    "tau/commonsense_qa",
    cache_dir=custom_cache_dir,
    split="train"          # CommonSense QA 有 train / validation，train 样本多
)

# 只保留答案合法的数据
ds = ds_dict.filter(lambda x: x["answerKey"] in {"A", "B", "C", "D", "E"})

SYSTEM_PROMPT = (
    "Answer the following multiple-choice commonsense question. "
    "Only output the letter of the correct choice, nothing else."
)

def csqa_to_text(example):
    q = example["question"].strip()
    choices = example["choices"]["text"]   # 列表长度=5
    answer_idx = ["A", "B", "C", "D", "E"].index(example["answerKey"])
    label = "ABCDE"[answer_idx]
    choice_str = " ".join([f"{c}) {text}" for c, text in zip("ABCDE", choices)])
    text = f"{SYSTEM_PROMPT}\nQuestion: {q}\n{choice_str}\nAnswer: {label}"
    return {"text": text}

print(">>> 格式化数据 …")
ds = ds.map(csqa_to_text)

def tokenize(sample):
    return tokenizer(
        sample["text"],
        padding=False,
        truncation=True,
        max_length=MAX_SEQUENCE_LENGTH,
        add_special_tokens=False
    )

ds = ds.map(tokenize, remove_columns=ds.column_names)

# 过滤 & 打乱
ds = ds.filter(lambda x: len(x["input_ids"]) >= 64)
ds = ds.shuffle(seed=42).select(range(min(NUM_CALIBRATION_SAMPLES, len(ds))))
print(f">>> 校准样本数: {len(ds)}")

# ---------------- 5. 量化 ----------------
print(">>> 开始 GPTQ 量化 …")
recipe = GPTQModifier(
    targets="Linear",
    scheme="W4A16",
    ignore=["lm_head"]
)

oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

# ---------------- 6. 保存结果 ----------------
print(">>> 保存量化模型 …")
SAVE_QUANT_DIR.mkdir(parents=True, exist_ok=True)
model.save_pretrained(SAVE_QUANT_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_QUANT_DIR)
print("✅ 量化完成，已保存到：", SAVE_QUANT_DIR)
