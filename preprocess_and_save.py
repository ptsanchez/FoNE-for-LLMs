# preprocess_and_save.py
import re
import os
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# --- Model and Tokenizer Loading ---
print("Loading Qwen7B model for preprocessing...")
# Make sure to use the correct model name and settings
model_name = "Qwen/Qwen2-7B-Instruct" 
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
print("Model loaded successfully.")

# --- Generation Function (from your code) ---
def generate_equation(question):
    prompt_template = (
        "Your task is to act as a mathematical translator. Convert the following natural language word problem into a single, solvable arithmetic expression. Do not solve the problem, only provide the equation. Your final response should only include numbers and relevant mathematical symbols and operations.\n"
        "Do not include any spaces in your final mathematical equation. An example of a good equation is: (16-3-4)*2"
        "Problem: {question}"
    )
    message = [
        {"role": "user", "content": prompt_template.format(question=question)},
    ]
    tokenized = tokenizer.apply_chat_template(
        message, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    
    outputs = model.generate(
        tokenized,
        max_new_tokens=100, # Reduced for efficiency
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    
    # Correctly decode only the newly generated tokens
    input_length = tokenized.shape[1]
    response_text = tokenizer.decode(outputs[0, input_length:], skip_special_tokens=True)
    return response_text

# --- Main Transformation Function ---
def transform_question(example):
    """
    Takes a single example from the dataset, generates the equation,
    and replaces the 'question' field.
    """
    new_question = generate_equation(example['question'])
    example['question'] = new_question
    return example

# --- Main Execution Logic ---
if __name__ == "__main__":
    # Define the directory where you'll save the data
    save_path = "./gsm8k_preprocessed"

    # Check if the data already exists to avoid re-running
    if os.path.exists(save_path):
        print(f"Preprocessed data already exists at {save_path}. Skipping generation.")
    else:
        print("Loading original gsm8k dataset...")
        original_dataset = load_dataset("openai/gsm8k", "main")
        
        print("Starting the transformation process (this will take a long time)...")
        # Use .map() with a progress bar. Tqdm is integrated automatically.
        modified_dataset = original_dataset.map(transform_question)
        
        print(f"Saving the processed dataset to {save_path}...")
        modified_dataset.save_to_disk(save_path)
        print("Dataset saved successfully!")

    print("\nPreprocessing is complete. You can now run your main training script.")
