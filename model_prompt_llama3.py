import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)

# =========================
# CONFIG
# =========================

#MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
MODEL_ID = "D:/hf_models/LLaMA3"
USE_4BIT = True        # set True if VRAM < 12GB
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.7
TOP_P = 0.9

# =========================
# CUDA CHECK
# =========================

if not torch.cuda.is_available():
    raise RuntimeError("CUDA not available. Install CUDA-enabled PyTorch.")

device = "cuda"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

print("CUDA device:", torch.cuda.get_device_name(0))

# =========================
# TOKENIZER
# =========================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    use_fast=False
)

# =========================
# MODEL LOADING
# =========================

if USE_4BIT:
    print("Loading model in 4-bit mode...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        quantization_config=bnb_config,
        use_safetensors=True
    )
else:
    print("Loading model in FP16 mode...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        use_safetensors=True
    )

model.eval()

# =========================
# PROMPT (LLaMA-3 FORMAT)
# =========================

def build_prompt(user_input: str) -> str:
    return f"""<|begin_of_text|>
<|system|>
You are a helpful AI assistant.
<|end_of_system|>
<|user|>
{user_input}
<|end_of_user|>
<|assistant|>
"""

# =========================
# GENERATION FUNCTION
# =========================

@torch.no_grad()
def generate_response(user_input: str) -> str:
    prompt = build_prompt(user_input)

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(model.device)

    output = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id
    )

    return tokenizer.decode(
        output[0],
        skip_special_tokens=True
    )

# =========================
# MAIN LOOP
# =========================

if __name__ == "__main__":
    print("\nLLaMA-3 Chat (type 'exit' to quit)\n")

    while True:
        user_text = input("You: ")
        if user_text.lower() in {"exit", "quit"}:
            break

        response = generate_response(user_text)
        print("\nAssistant:\n", response)
        print("-" * 80)
