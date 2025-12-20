import os
import json
import torch
import logging
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,  
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,  
    set_seed,
)
from peft import LoraConfig
from trl import SFTTrainer
from huggingface_hub import login
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

# Login to Hugging Face Hub
# Ensure you have a valid token in your .env file
if HUGGINGFACE_TOKEN:
    login(token=HUGGINGFACE_TOKEN)
else:
    print("Hugging Face token not found. Please set it in your .env file.")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# --- Configuration ---
MODEL_ID = "google/gemma-3-1b-it"
DATASET_DIR = "./preprocessed_jsonl"  # Directory containing .jsonl files (preprocessed files)
OUTPUT_DIR = "./results"
NEW_MODEL_NAME = "gemma3-1b-contract-analyzer"

# --- Training Parameters ---
BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 4
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3
MAX_SEQ_LENGTH = 2048


def format_dataset(sample):
    """
    Formats a single data sample into the Gemma chat format.
    This function will be used by SFTTrainer's `formatting_func`.
    """
    instruction = (
        "You are a smart contract security analyzer. Analyze the following "
        "Solidity code and provide analysis in JSON format with risk_level, "
        "summary, and key_terms_operations."
    )

    code = sample.get("original_clause_or_definition", "").strip()

    response_data = {
        "risk_level": sample.get("risk_level", "N/A"),
        "summary": sample.get("summary", "N/A"),
        "key_terms_operations": sample.get("key_terms_operations", []),
    }
    response_json = json.dumps(response_data, indent=2)

    # Gemma's chat format requires a specific structure with turns
    formatted_text = f"""<start_of_turn>user
{instruction}

Analyze this Solidity code:
```solidity
{code}
```<end_of_turn>
<start_of_turn>model
{response_json}<end_of_turn>"""

    return formatted_text


def setup_model_and_tokenizer():
    """Sets up the model and tokenizer for training."""
    logger.info(f"Loading base model and tokenizer from: {MODEL_ID}")

    # Quantization configuration for loading the model in 4-bit
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Set tokenizer's pad_token to its eos_token.")

    # Load the model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation='eager', 
    )

    return model, tokenizer


def train():
    """Main function to run the training pipeline."""
    set_seed(42)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load Model and Tokenizer
    model, tokenizer = setup_model_and_tokenizer()

    # 2. Load and Prepare Dataset
    logger.info("Loading and preparing dataset...")
    
    jsonl_files = [
        os.path.join(DATASET_DIR, f)
        for f in os.listdir(DATASET_DIR)
        if f.endswith(".jsonl")
    ]

    if not jsonl_files:
        raise FileNotFoundError(f"No .jsonl files found in the directory: {DATASET_DIR}")

    dataset = load_dataset("json", data_files=jsonl_files, split="train")
    logger.info(f"Loaded {len(dataset)} samples from {len(jsonl_files)} file(s).")
    
    
    logger.info(f"Sample formatted text:\n{format_dataset(dataset[0])[:350]}...")

    # 3. Configure PEFT (LoRA)
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 4. Configure Training Arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        fp16=True, 
        logging_steps=10,
        save_steps=100,
        save_strategy="steps",
        optim="paged_adamw_8bit", # Optimizer suitable for quantized models
        lr_scheduler_type="cosine",
        warmup_steps=10,
        gradient_checkpointing=True,
        save_total_limit=2,
        report_to="none", 
        push_to_hub=False,
    )

   
    # SFTTrainer is ideal for this task as it simplifies the process of
    # supervised fine-tuning on formatted prompt-response pairs.
    logger.info("Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
        formatting_func=format_dataset, 
    )

    # 6. Start Training
    logger.info("Starting model training...")
    trainer.train()
    logger.info("Training complete.")

    # 7. Save the Final Model and Tokenizer
    logger.info(f"Saving final model adapter to {NEW_MODEL_NAME}...")
    trainer.save_model(NEW_MODEL_NAME)
    tokenizer.save_pretrained(NEW_MODEL_NAME)
    logger.info("Model and tokenizer saved successfully.")


def test_model():
    """Loads the fine-tuned model and tests it with a sample prompt."""
    logger.info(f"Testing trained model from: {NEW_MODEL_NAME}")

    # Load the fine-tuned model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(NEW_MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        NEW_MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # Create a test prompt in the same format used for training
    test_code = """
function transfer(address to, uint256 amount) public {
    require(balances[msg.sender] >= amount, "Insufficient balance");
    balances[msg.sender] -= amount;
    balances[to] += amount;
}
    """.strip()
    
    instruction = (
        "You are a smart contract security analyzer. Analyze the following "
        "Solidity code and provide analysis in JSON format with risk_level, "
        "summary, and key_terms_operations."
    )

    
    test_prompt = f"""<start_of_turn>user
{instruction}

Analyze this Solidity code:
```solidity
{test_code}
```<end_of_turn>
<start_of_turn>model
"""

    inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
    
    logger.info("Generating response...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=400,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    # Clean up the output to show only the generated part
    cleaned_response = response.split("<start_of_turn>model\n")[-1]
    
    logger.info("--- Model Response ---")
    print(cleaned_response)
    logger.info("--- End of Response ---")


if __name__ == "__main__":
    train()
    