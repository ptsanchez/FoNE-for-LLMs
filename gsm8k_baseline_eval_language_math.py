import torch
import re
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, logging
from datasets import load_dataset
import time
logging.set_verbosity_error()

def extract_final_answer(model_output: str) -> float | None:
    """
    Parses the model's text output to find the final numerical answer.
    For GSM8K answers, '#### <number>'.
    """
    match = re.search(r"####\s*([\d,]+\.?\d*)", model_output)
    if match:
        final_answer_str = match.group(1).replace(",", "")
        try:
            return float(final_answer_str)
        except ValueError:
            return None

    # as fallback, try to find the last number in the string
    all_numbers = re.findall(r"[\d,]+\.?\d*", model_output)
    if all_numbers:
        final_answer_str = all_numbers[-1].replace(",", "")
        try:
            return float(final_answer_str)
        except ValueError:
            return None

    return None


def main():
    start_time = time.time()

    MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
    DATASET_ID = "Onlydrinkwater/language_math_multiplication_10base"
    OUTPUT_CSV = "/home/ubuntu/FoNE-for-LLMs/results/gsm8k_base_model_math_language_results.csv"

    # Load the model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    print("Model and tokenizer loaded")

    # Number of tests to be evaluated, set to None for full test set evaluation
    NUM_SAMPLES = None
    print(f"Loading dataset: {DATASET_ID}")
    dataset = load_dataset(DATASET_ID)
    if NUM_SAMPLES:
        test_set = dataset['test'].select(range(NUM_SAMPLES))
    else:
        test_set = dataset['test']
    print(f"Dataset loaded. Evaluating on {len(test_set)} samples.")

    correct_predictions = 0
    results = []

    prompt_template = (
        "Please solve the following math problem step by step."
        "After providing the reasoning, the final answer must be stated in the following format: #### <answer>. <answer> is only the final numerical result.\n\n"
        "Question: {question}"
    )

    print('Start evaluating ...')
    with torch.inference_mode():
        for example in tqdm(test_set, desc="Evaluating model"):
            question = example['question']
            ground_truth_answer = example['label']

            messages = [
                {"role": "user", "content": prompt_template.format(question=question)}
            ]
            tokenized_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(model.device)

            # Generate the model's response
            outputs = model.generate(
                tokenized_prompt,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

            response_text = tokenizer.decode(outputs[0][tokenized_prompt.shape[1]:], skip_special_tokens=True)
            predicted_answer = extract_final_answer(response_text)

            is_correct = False
            if predicted_answer is not None:
                # small tolerance for float comparison
                if abs(predicted_answer - ground_truth_answer) < 1e-4:
                    correct_predictions += 1
                    is_correct = True

            results.append({
                "question": question,
                "ground_truth": ground_truth_answer,
                "model_output": response_text,
                "predicted_answer": predicted_answer,
                "is_correct": is_correct
            })
            if (len(results) == 1):
                print(question)
                print(predicted_answer)
                print(is_correct)
            if len(results) % 200 == 0:
                print(f"Processed {len(results)} samples so far...")

    accuracy = (correct_predictions / len(test_set)) * 100
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    end_time = time.time()
    total_hours = (end_time - start_time) / 3600

    print('Evaluation complete')
    print('Results:')
    print(f'Model: {MODEL_ID}')
    print(f'Dataset: {DATASET_ID}')
    print(f'Total Samples: {len(test_set)}')
    print(f'Correct Predictions: {correct_predictions}')
    print(f'Accuracy: {accuracy:.2f}%')
    print(f'Total Time Used: {total_hours:.2f} hours')
    print(f'Saved to: {OUTPUT_CSV}')


if __name__ == "__main__":
    main()
