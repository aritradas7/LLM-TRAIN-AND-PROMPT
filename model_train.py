import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

# ======================
# CONFIG
# ======================
MODEL_ID = "D:/hf_models/LLaMA3"
DATASET_PATH = "./solutions/sql_server_dba_instruction_dataset_30000.jsonl"
OUTPUT_DIR = "D:/hf_models/LLaMA3-SQLDBA"

# ======================
# QUANTIZATION (QLoRA)
# ======================
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# ======================
# TOKENIZER (IMPORTANT FOR LLAMA-3)
# ======================
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    use_fast=True
)

# LLaMA-3 does NOT have a pad token
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
tokenizer.model_max_length = 1024


# ======================
# MODEL
# ======================
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    dtype=torch.bfloat16
)

model.config.use_cache = False  # REQUIRED for training

# ======================
# LOAD DATASET (JSONL)
# ======================
dataset = load_dataset(
    "json",
    data_files=DATASET_PATH,
    split="train"
)

# ======================
# PROMPT FORMAT (ORCA STYLE)
# ======================
def format_prompt(example):
    return f"""### Instruction:
{example['instruction']}

### Input:
{example['input']}

### Response:
Explanation:
{example['response']['explanation']}

SQL Solution:
{example['response']['sql_code']}
"""

# ======================
# TRAINING CONFIG
# ======================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=False,
    bf16=True,                     # LLaMA-3 prefers bf16
    logging_steps=50,
    save_strategy="epoch",
    report_to="none"
)

# ======================
# LoRA CONFIG (LLAMA-3)
# ======================
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# ======================
# TRAINER
# ======================
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    args=training_args,
    formatting_func=format_prompt,
    processing_class=tokenizer
)

trainer.train()
trainer.save_model(OUTPUT_DIR)
