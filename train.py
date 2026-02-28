# Torch stuff
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import numpy as np
# Other files stuff
from dataset import BilingualDataset, load_data
from model import get_model
from config import get_weights_file_path, get_latest_weights, get_config
from test import run_validation, run_validation_teacher_forcing, run_validation_visualization, run_test, answer, run_full_validation
import os

# HuggingFace stuff
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace, CharDelimiterSplit, Punctuation
from datasets import Dataset as HFDataset

# Metrics stuff
import warnings

# Easy access stuff
from pathlib import Path
from tqdm import tqdm

# Set the random seed for this project, for reproducibility.
import random
SEED = get_config()["seed"]
torch.manual_seed(SEED)
random.seed(SEED)

            
def get_all_sentences(
        dataset: HFDataset,
    ):
    """
    Yields elements of the provided dataset.

    Args:
        dataset (HFDataset): Dataset to iterate through.
        language (str): Language given as a language code, present in the dataset.

    Yields:
        str: Sentence from the dataset in the provided language.
    """
    for item in dataset:
        yield item['riddle']['Riddle'] + ' ' + item['riddle']['Answer'] + ' ' + item['riddle']['Hint']
        

def get_or_build_tokenizer(
        config, 
        dataset: HFDataset, 
        force_rewrite: bool = False,
        min_frequency: int = 1,
        vocab_size: int = 1000000
    ) -> Tokenizer:
    """ 
    If the path to tokenizer file is not specified in the config, or if
    we force rewrite, then build a tokenizer from scratch.
    Else, get the tokenizer from the specified file.

    Args:
        config: A config file.
        dataset (HFDataset): HuggingFace dataset of translations to build the tokenizer from.
        language (str): Language from the dataset for the tokenizer.
        force_rewrite (bool): If the function should disregard the config file.
        min_frequency (int): Minimum frequency of a word in the dataset to add it to the vocabulary.
        vocab_size (int): Maximum size of the vocabulary.

    Returns:
        Tokenizer: A tokenizer for the specified language built from a vocabulary formed by sentences from the dataset.
    """
    # Get the path from config.
    tokenizer_path = Path(config['tokenizer_file'])

    tokenizer = Tokenizer.from_pretrained('t5-small')
    tokenizer.save(str(tokenizer_path))

    # Return the tokenizer and print the number of tokens in it.
    print(f"Number of tokens for trained tokenizer is {tokenizer.get_vocab_size}.")
    return tokenizer


def get_dataset(config):
    """
    Initializes the training and validation datasets.
    Initializes the tokenizers.

    Args:
        config: A config file.

    Returns:
        DataLoader: Training dataset dataloader.
        DataLoader: Validation dataset dataloader.
        Tokenizer: Source language tokenizer.
        Tokenizer: Target language tokenizer.
    """
    # Load the data
    dataset_raw = load_data()

    # Initialize the training, validation and test dataset sizes.
    dataset_size = len(dataset_raw)
    train_dataset_size = int(0.9 * dataset_size)
    validation_dataset_size = int(0.08 * dataset_size)
    test_dataset_size = dataset_size - train_dataset_size - validation_dataset_size

    # Split the data into datasets.
    training_dataset_raw, validation_dataset_raw, test_dataset_raw = random_split(dataset_raw, [train_dataset_size, validation_dataset_size, test_dataset_size])

    tokenizer = get_or_build_tokenizer(config, training_dataset_raw, force_rewrite = True)

    # Define the BilingualDataset objects for the training and validation datasets.
    training_dataset = BilingualDataset(training_dataset_raw, tokenizer, config['context_size'])
    validation_dataset = BilingualDataset(validation_dataset_raw, tokenizer, config['context_size'])
    test_dataset = BilingualDataset(test_dataset_raw, tokenizer, config['context_size'])

    # Calculate the maximum lengths of the sentences in training dataset.
    # Only for testing purposes.
    max_len_source = 0
    max_len_target = 0
    for item in dataset_raw:
        source_ids = tokenizer.encode(item['riddle']['Riddle']).ids
        target_ids = tokenizer.encode(item['riddle']['Answer']).ids
        max_len_source = max(max_len_source, len(source_ids))
        max_len_target = max(max_len_target, len(target_ids))

    print(f"Max length of source sentence: {max_len_source}")
    print(f"Max length of target sentence: {max_len_target}")

    # Define the DataLoader objects for training and validation datasets.
    training_dataloader = DataLoader(training_dataset, batch_size = config['batch_size'], shuffle = True, num_workers=4)
    validation_dataloader = DataLoader(validation_dataset, batch_size = 1, shuffle = True, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size = 1, shuffle = True, num_workers=4)

    return training_dataloader, validation_dataloader, test_dataloader, tokenizer


def train_model(config):
    """
    Train the transformer model with the given parameters.

    Args:
        config: A config file.
    """
    # Use cuda if possible, otherwise use cpu.
    def get_device():
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    # usage
    device = get_device()
    print(f"Training running on: {device}")

    # Make the folder for the model weights.
    Path(config['model_folder']).mkdir(parents = True, exist_ok = True)

    # Get the datasets and define the model.
    training_dataloader, validation_dataloader, test_dataloader, tokenizer = get_dataset(config)
    m = get_model(config, tokenizer.get_vocab_size()).to(device)
    model = torch.compile(m)
    print(model)
    t = 0
    for n, p in model.named_parameters():
        t += p.numel()
    print(f"Total parameters: {t}")

    # Initialize the writer to visualize data.
    writer = SummaryWriter(config['experiment_name'])
    print(f"TensorBoard logs are being saved to: {os.path.abspath(config['experiment_name'])}")

    # Initialize the Adam optimizer.
    # Adjusts the learning rate as the model trains.
    optimizer = torch.optim.Adam(model.parameters(), lr = config['learning_rate'], eps = 1e-9)

    total_steps = config['num_epochs'] * len(training_dataloader)

    # Load a pretrained model if defined and if it exists.
    initial_epoch = 0
    global_step = 0
    preload = config['preload']
    model_filename = get_latest_weights(config) if preload == 'latest' else get_weights_file_path(config, preload) if preload else None

    if model_filename:
        print(f"Preloading model {model_filename}.")
        state = torch.load(model_filename)
        optimizer.load_state_dict(state['optimizer_state_dict'])
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        global_step = state['global_step']
    else:
        print("No model to preload, starting from the beginning.")

    # Define the loss function.
    loss_function = nn.CrossEntropyLoss(ignore_index = tokenizer.token_to_id('<pad>'), label_smoothing = 0.1).to(device)

    # Run the epochs.
    P_schedule = np.linspace(0,1, config['num_epochs'])
    for epoch in range(initial_epoch, config['num_epochs']):
        training_dataloader.dataset.change_P(P_schedule[epoch])
        # Create a batch iterator and iterate through the batches.
        batch_iterator = tqdm(training_dataloader, desc = f"Processing epoch {epoch:02d}")
        for batch in batch_iterator:

            # Put the model in the training state.
            model.train()

            # Move the tensors to the device.
            encoder_input = batch['encoder_input'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            decoder_mask = batch['decoder_mask'].to(device)

            # Calculate the outputs of the model for this batch.
            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
            transformer_output = model.project(decoder_output)

            label = batch['label'].to(device)

            # Calculate the loss, comparing the output of the model to the expected output (label)
            loss = loss_function(transformer_output.view(-1, tokenizer.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            # Add loss to the tensorboard.
            writer.add_scalar('train_loss', loss.item(), global_step)
            writer.flush()

            # Calculate the gradient.
            loss.backward()

            # Adjust the learning rate.
            optimizer.step()
            optimizer.zero_grad()

            # Adjust the global step.
            global_step += 1

        # Run the validation at the end of every epoch.
        #run_validation_visualization(model, validation_dataloader, source_tokenizer, target_tokenizer, config['context_size'], device, lambda msg: batch_iterator.write(msg), writer, global_step, number_examples = 1)
        # run_validation_teacher_forcing(model, validation_dataloader, loss_function, target_tokenizer, device)
        #run_validation(model, test_dataloader, tokenizer, 100, device, writer, len(training_dataloader)*epoch)
        run_full_validation(model, validation_dataloader, tokenizer, config['context_size'], device, writer, epoch*config['batch_size'], loss_function, 0.0)

        # Save weights at certain 'milestone' epochs.
        model_filename = get_weights_file_path(config, f'{epoch:02d}')
        if epoch % 10 == 9 or epoch == 0 or epoch == config['num_epochs'] - 1:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'global_step': global_step
            }, model_filename)
            
    # Run the validation at the end of training.
    #run_validation_visualization(model, validation_dataloader, source_tokenizer, target_tokenizer, config['context_size'], device, lambda msg: batch_iterator.write(msg), writer, global_step, number_examples = 50)
    #run_test(test_dataloader)

    print('-'*80)
    print("Final model test without hints")
    run_full_validation(model, test_dataloader, tokenizer, config['context_size'], device, None,
                        0, loss_function, 0.0)
    print('-' * 80)
    print("Final model test with hints")
    run_full_validation(model, test_dataloader, tokenizer, config['context_size'], device, None,
                        0, loss_function, 1.0)
    print('-' * 80)
    writer.close()



def dataset_test(config):
    training_dataloader, validation_dataloader, test_dataloader, tokenizer = get_dataset(config)
    print(len(training_dataloader))
    print(next(iter(training_dataloader)))
    print(next(iter(validation_dataloader)))
    print(next(iter(test_dataloader)))
    print(tokenizer.get_vocab_size())

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    config = get_config()
    train_model(config)


    #dataset_test(get_config())
