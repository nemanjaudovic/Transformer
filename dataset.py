import torch

from torch.utils.data import Dataset as TorchDataset
from datasets import Dataset as HFDataset
from tokenizers import Tokenizer
# from translate.storage.tmx import tmxfile
import pandas as pd

from typing import Any, Dict
import random


# class RiddleDataset(TorchDataset):
#     def __init__(self, path: str, tokenizer: Tokenizer, context_size: int):
#         self.dataset_file = pd.read_csv(path)
#         self.tokenizer = tokenizer
#         self.sos_token = torch.tensor([tokenizer.token_to_id('[SOS]')], dtype=torch.int64)
#
#         self.eos_token = torch.tensor([tokenizer.token_to_id('[EOS]')], dtype=torch.int64)
#
#         self.pad_token = torch.tensor([tokenizer.token_to_id('[PAD]')], dtype=torch.int64)
#
#
#     def __getitem__(self, idx):
#         pass
#
#     def __len__(self):
#         pass

class BilingualDataset(TorchDataset):
    """
    Wrapper class of Torch Dataset.
    Has to have methods __init__, __len__ and __getitem__ to function properly.
    """

    def __init__(
            self, 
            dataset: HFDataset, 
            tokenizer: Tokenizer,
            context_size: int,
            P: float = 1.0
        ) -> None:
        """Initializing the BilingualDataset object.

        Args:
            dataset (HFDataset): 
                HuggingFace dataset with columns id and translations.
                    id is the number of the current row.
                    translations is a dictionary with entries 'language': sentence,
                    which has at least two different languages.
            source_tokenizer (Tokenizer): Tokenizer for the source language.
            target_tokenizer (Tokenizer): Tokenizer for the target language.
            source_language (str): Source language for the translations.
            target_language (str): Target language for the translations.
            context_size (int): Maximum allowed length of a sentence (in either language).
        """
        super().__init__()

        # Initializing context size.
        self.context_size = context_size

        # Initializing the dataset.
        self.dataset = dataset

        # Initializing the tokenizers.
        self.tokenizer = tokenizer

        # Set hint including probability
        self.P = P

        # Initializing the start of sentence, end of sentence and padding tokens.
        
        # Start of sentence token signifies the beginning of a sentence.
        self.sos_token = torch.tensor([tokenizer.token_to_id('[SOS]')], dtype = torch.int64)
        
        # End of sentence token signifies the end of a sentence.
        self.eos_token = torch.tensor([tokenizer.token_to_id('[EOS]')], dtype = torch.int64)

        # Padding token signifies the placeholder token for sentences shorter than context size, which fills the empty spaces.
        self.pad_token = torch.tensor([tokenizer.token_to_id('[PAD]')], dtype = torch.int64)


    def change_P(self, new_P):
        self.P = new_P

    def build_prompt(self, riddle, hint):
        # P -> prob that hint is included
        if random.random() > self.P:
            hint = ""

        return f"riddle: {riddle}, hint: {hint}"


    def __len__(self) -> int:
        """
        Returns:
            int: Number of sentences in the dataset.
        """
        return len(self.dataset)
    
    
    def __getitem__(
            self, 
            index: int
        ) -> Dict[str, Any]:
        """Gets the row from the dictionary at a specified index.

        Args:
            index (int): Index at which to return the element from the list.

        Raises:
            ValueError: _description_

        Returns:
            Dict[str, Any]: A dictionary with 7 fields:
                encoder_input: 
                    Input to be fed to the encoder. 
                    Tensor of dimension (context_size)
                decoder_input:
                    Input to be fed to the decoder. 
                    Tensor of dimension (context_size)
                encoder_mask:
                    Mask for the encoder, that will mask any padding tokens.
                    Tensor of dimension (1, 1, context_size)
                decoder_mask:
                    Mask for the decoder, that will mask any padding tokens and won't allow predictions in the past.
                    Tensor of dimension (1, context_size, context_size)
                label:
                    Expected model output.
                    Tensor of dimension (context_size)
                source_text:
                    Sentence in the source language.
                target_text:
                    Sentence in the target language.
        """
        # Get the index-th row of the dataset.
        source_target_pair = self.dataset[index]

        # Get the sentence in the source and target language.
        riddle_text = source_target_pair['riddle']['Riddle'].lower()
        hint_text = source_target_pair['riddle']['Hint'].lower()
        source_text = self.build_prompt(riddle_text, hint_text)
        target_text = source_target_pair['riddle']['Answer'].lower()

        # Number of tokens in sentences.
        encoder_input_tokens = self.tokenizer.encode(source_text).ids
        decoder_input_tokens = self.tokenizer.encode(target_text).ids

        # Number of padding tokens for both sentences.
        # Encoder already has len(encoder_input_tokens), SOS and EOS.
        encoder_num_padding_tokens = self.context_size - len(encoder_input_tokens) - 2
        # Decoder already has len(decoder_input_tokens), and:
        #       SOS token for the input;
        #       EOS token for the label.
        decoder_num_padding_tokens = self.context_size - len(decoder_input_tokens) - 1
        
        # Make sure the sentence isn't too long in either language.
        if encoder_num_padding_tokens < 0 or decoder_num_padding_tokens < 0:
            raise ValueError("Sentence is too long!")
        
        # Encoder input is [SOS] token_enc[1] token_enc[2] ... token_enc[K] [EOS] [PAD] [PAD] ... [PAD].
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(encoder_input_tokens, dtype = torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * encoder_num_padding_tokens, dtype = torch.int64)
            ],
            dim = 0
        )

        # Decoder input is [SOS] token_dec[1] token_dec[2] ... token_dec[J] [PAD] [PAD] ... [PAD].
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(decoder_input_tokens, dtype = torch.int64),
                torch.tensor([self.pad_token] * decoder_num_padding_tokens, dtype = torch.int64)
            ],
            dim = 0
        )

        # Label is token_dec[1] token_dec[2] ... token_dec[J] [EOS] [PAD] [PAD] ... [PAD].
        label = torch.cat(
            [
                torch.tensor(decoder_input_tokens, dtype = torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * decoder_num_padding_tokens, dtype = torch.int64)
            ],
            dim = 0
        )

        # Make sure the tensor dimensions are correct.
        assert encoder_input.size(0) == self.context_size
        assert decoder_input.size(0) == self.context_size
        assert label.size(0) == self.context_size

        # Return the appropriate values.
        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0)),
            "label": label,
            "source_text" : source_text,
            "target_text" : target_text
        }

    
def causal_mask(size: int) -> torch.Tensor:
    """
    Generates a causal mask for the decoder. This is a triangular matrix that
    has all ones as inputs which deals with decoder having access to words that
    have not yet been translated.

    Args:
        size (int): Size of the mask matrix.

    Returns:
        torch.Tensor: Triangular matrix with all ones, of dimension (1, size, size).
    """
    mask = torch.triu(torch.ones(1, size, size), diagonal = 1).type(torch.int)
    return mask == 0


def load_data() -> HFDataset:
    """
    Translates the .tmx file into a HFDataset with given languages.

    Args:
        source_language (str): Original language of the dataset.
        target_language (str): Translated language of the dateset.

    Returns:
        HFDataset: HuggingFace dataset with columns id and translations.
                    id - the number of the current row.
                    translations - a dictionary with entries 'language': sentence,
                    which has at least two different languages.
    """
    # Open the tmx file.
    df = pd.read_csv('Riddles.csv')

    # Define the data in the HuggingFace standard.
    data = {'id' : [], 'riddle': []}
    i = 0

    # Iterate through the file and add the rows one by one.
    for _, row in df.iterrows():

        data["id"].append(str(i))
        i = i + 1

        data["riddle"].append({f"Riddle": row['Riddle'][:-1], f"Hint": row['Hint'], f"Answer": row['Answer']})

    # Define the HuggingFace standard dataset.
    dataset = HFDataset.from_dict(data)

    return dataset

if __name__ == '__main__':
    pass
