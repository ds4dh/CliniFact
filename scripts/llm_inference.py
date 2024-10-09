import torch
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import bitsandbytes as bnb
import argparse
from huggingface_hub import login
from datasets import load_dataset
import pandas as pd
from sklearn.metrics import classification_report
from peft import PeftModel, PeftConfig

# Log in to Hugging Face Hub
login(token="your_token_here")

@dataclass
class ScriptArguments:
    model_name: str
    load_in_8bit: bool
    load_in_4bit: bool
    trust_remote_code: bool
    token: str
    test_dataset_name: str
    dataset_text_field: str
    max_new_tokens: int
    cache_dir: str

def load_model_and_tokenizer(args):
    if args.load_in_8bit and args.load_in_4bit:
        raise ValueError("You can't load the model in 8 bits and 4 bits at the same time")
    elif args.load_in_8bit or args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=args.load_in_8bit,
            load_in_4bit=args.load_in_4bit,
            bnb_4bit_use_double_quant=args.load_in_4bit,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        device_map = "auto"
        torch_dtype = torch.bfloat16
    else:
        device_map = None
        quantization_config = None
        torch_dtype = None

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=quantization_config,
        device_map=device_map,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=torch_dtype,
        token=args.token,
        cache_dir=args.cache_dir
    )

    if args.use_trained_model:
        if args.model_path:
            model = PeftModel.from_pretrained(base_model, args.model_path)
        else:
            raise ValueError("Model path must be provided when using the trained model")
    else:
        model = base_model

    # Your code for using the model goes here
    print("Model loaded successfully")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, 
        trust_remote_code=args.trust_remote_code, 
        token=args.token,
        cache_dir=args.cache_dir
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left" # inference

    return model, tokenizer

def generate_text(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)

    with torch.no_grad():
        outputs = model(**inputs)
        next_token_logits = outputs.logits[0, -1, :]
        next_token_probs = torch.softmax(next_token_logits, -1)
        label_indices = tokenizer.convert_tokens_to_ids(["TRUE", "FALSE", "NONE"])
        output_prob = {
            'TRUE': next_token_probs[label_indices[0]].item(),
            'FALSE': next_token_probs[label_indices[1]].item(),
            'NONE': next_token_probs[label_indices[2]].item()
        }

        # Find the label with the highest probability
        max_label = max(output_prob, key=output_prob.get)
      
        return max_label

def main(args):
    model, tokenizer = load_model_and_tokenizer(args)

    # Load the test dataset
    test_dataset = load_dataset("json", data_files=args.test_dataset_name)

    reverse_label_mapping = {"TRUE": 1, "FALSE": 0, "NONE": 2}

    predictions = []
    # Perform inference on the test dataset
    for sample in test_dataset['train']:
        prompt = sample[args.dataset_text_field]
        prediction = generate_text(model, tokenizer, prompt)
        predictions.append(prediction)

    # Convert text predictions to numeric predictions
    numeric_predictions = [reverse_label_mapping.get(item.strip(), -1) for item in predictions]
    print(numeric_predictions)

    # Check if there are any unknown labels
    if -1 in numeric_predictions:
        print("Warning: Some predictions could not be mapped to numeric labels.")

    df = pd.read_csv("./data/processed/primary_outcome_publication_dataset/test_set.csv")

    labels = df['label'].tolist()
    labels = [int(item) for item in labels]

    print(classification_report(labels, numeric_predictions))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", default="meta-llama/Meta-Llama-3-70B", type=str)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--load_in_8bit", default=False, action="store_true")
    parser.add_argument("--load_in_4bit", default=True, action="store_true")
    parser.add_argument("--trust_remote_code", default=True, action="store_true")
    parser.add_argument("--token", default=None, type=str)
    parser.add_argument("--test_dataset_name", default="./data/processed/primary_outcome_publication_instruction/test_set.jsonl", type=str)
    parser.add_argument("--dataset_text_field", default="text", type=str)
    parser.add_argument("--max_new_tokens", default=20, type=int)
    parser.add_argument("--cache_dir", default="./model", type=str)
    parser.add_argument('--use_trained_model', action='store_true', help='Use the trained model')

    args = parser.parse_args()

    main(args)