import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import warnings
import os

from dataset import BilingualDataset, load_data
from model import get_model
from config import get_config
# Import the inference functions we created
from test import collect_all_predictions, predict_sequence

from tokenizers import Tokenizer

import random

config = get_config()
SEED = config["seed"]
torch.manual_seed(SEED)
random.seed(SEED)


def get_or_build_tokenizer():
    tokenizer = Tokenizer.from_pretrained('t5-small')
    print(f"Number of tokens for trained tokenizer is {tokenizer.get_vocab_size}.")
    return tokenizer


def get_test_dataset(config):
    """
    Recreates the splits exactly as they were during training 
    to isolate the Test Set.
    """
    dataset_raw = load_data()

    dataset_size = len(dataset_raw)
    train_dataset_size = int(0.9 * dataset_size)
    validation_dataset_size = int(0.08 * dataset_size)
    test_dataset_size = dataset_size - train_dataset_size - validation_dataset_size

    _, _, test_dataset_raw = random_split(dataset_raw, [train_dataset_size, validation_dataset_size, test_dataset_size])

    tokenizer = get_or_build_tokenizer()

    test_dataset = BilingualDataset(test_dataset_raw, tokenizer, config['context_size'])

    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

    return test_dataloader, tokenizer


def run_testing_pipeline(config):
    """
    Main function to load model and run full tests.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 1. Prepare Data
    print("Preparing Test Data...")
    test_dataloader, tokenizer = get_test_dataset(config)
    print(f"Test Data loaded. Size: {len(test_dataloader)}")

    # 2. Load Model Architecture
    print("Initializing Model...")
    model = get_model(config, tokenizer.get_vocab_size()).to(device)

    # Optional: Compile if your saved weights expect it, 
    # but often safer to load weights first then compile for inference.
    # model = torch.compile(model) 

    # 3. Load Weights
    model_filename = config['model_path']  # Ensure this is set in your config.py

    if os.path.exists(model_filename):
        print(f"Loading weights from: {model_filename}")
        state = torch.load(model_filename, map_location=device)

        # Handle cases where weights were saved with 'model.' prefix or '_orig_mod.'
        # or if you saved the whole checkpoint dict vs just state_dict
        if 'model_state_dict' in state:
            state_dict = state['model_state_dict']
        else:
            state_dict = state

        # Load state dict
        # strict=False allows loading even if there are minor prefix mismatches (like _orig_mod)
        model.load_state_dict(state_dict, strict=False)
        print("Weights loaded successfully.")
    else:
        raise FileNotFoundError(f"Cannot find model file at {model_filename}")

    # 4. Run Full Prediction Loop
    print("Starting Inference on Test Set...")
    results = collect_all_predictions(
        model,
        test_dataloader,
        tokenizer,
        device,
        max_length=100
    )

    # 5. Save Results to CSV
    output_file = "test_results.csv"
    df = pd.DataFrame(results)

    # Simple accuracy check (exact match)
    # Note: For riddles, exact match is harsh, but good baseline
    df['Exact Match'] = df.apply(lambda x: x['True Answer'].strip().lower() == x['Predicted'].strip().lower(), axis=1)
    accuracy = df['Exact Match'].mean()

    print("\n" + "=" * 30)
    print(f"Testing Complete.")
    print(f"Exact Match Accuracy: {accuracy:.2%}")
    print(f"Results saved to {output_file}")
    print("=" * 30)

    df.to_csv(output_file, index=False)

    # 6. Sanity Check - Interactive Demo
    print("\nRunning a sanity check prediction:")
    sanity_q = "What has keys but can't open locks?"
    pred_text, _ = predict_sequence(model, tokenizer, sanity_q, device)
    print(f"Q: {sanity_q}")
    print(f"A: {pred_text}")


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    config = get_config()
    #config['model_path'] = ''

    run_testing_pipeline(config)