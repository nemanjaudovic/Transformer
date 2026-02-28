import torch
import os
from pathlib import Path
from tokenizers import Tokenizer

from model import get_model
from config import get_config
from test import _greedy_decode


def predict_riddle(riddle_text: str, model, tokenizer, device, max_length=50):
    """
    Takes a raw string, prepares the tensors, and uses your _greedy_decode
    to generate the answer.
    """
    model.eval()

    with torch.no_grad():
        sos_id = tokenizer.token_to_id('<pad>')
        eos_id = tokenizer.token_to_id('</s>')

        # Encode the string -> List of IDs
        input_ids = tokenizer.encode(riddle_text).ids

        input_ids = [sos_id] + input_ids + [eos_id]

        # Batch=1
        encoder_input = torch.tensor(input_ids).unsqueeze(0).to(device)

        # Create Mask (1, 1, 1, seq_len)
        pad_id = tokenizer.token_to_id('<pad>')
        encoder_mask = (encoder_input != pad_id).unsqueeze(0).unsqueeze(0).int().to(device)

        # Run Inference
        model_out_ids = _greedy_decode(
            model,
            encoder_input,
            encoder_mask,
            tokenizer,
            max_length,
            device
        )

        # Convert IDs back to String
        predicted_text = tokenizer.decode(model_out_ids.detach().cpu().numpy(), skip_special_tokens=True)

        return predicted_text

def get_or_build_tokenizer():
    tokenizer = Tokenizer.from_pretrained('t5-small')
    print(f"Number of tokens for trained tokenizer is {tokenizer.get_vocab_size}.")
    return tokenizer

def main():
    # --- Setup ---
    config = get_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running inference on: {device}")

    # Load tokenizer
    tokenizer = get_or_build_tokenizer()

    # Load model
    model = get_model(config, tokenizer.get_vocab_size()).to(device)

    # Load Weights
    model_path = config['model_path']
    if os.path.exists(model_path):
        print(f"Loading weights from {model_path}...")
        state = torch.load(model_path, map_location=device)

        # Handle different saving formats (dict vs state_dict)
        model.load_state_dict(state['model_state_dict'])
    else:
        print("WARNING: No weights found. The model will output random garbage.")

    # --- Run Prediction ---
    test_riddle = "What has keys but can't open locks?"

    print("\n" + "=" * 40)
    print(f"Question: {test_riddle}")

    answer = predict_riddle(test_riddle, model, tokenizer, device)

    print(f"Answer:   {answer}")
    print("=" * 40 + "\n")


if __name__ == "__main__":
    main()