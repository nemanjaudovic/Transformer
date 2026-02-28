from typing import List, Tuple, Callable
from config import get_config
from model import Transformer, get_model
from dataset import Tokenizer, causal_mask
import torch
import os
from torch.utils.data import DataLoader
import torchmetrics
import tabulate
from tqdm import tqdm
import torch.nn as nn
import evaluate

from torch.utils.tensorboard import SummaryWriter


def compute_f1_score(predicted_text: str, target_text: str) -> float:
    """
    Calculates token-level F1 score (Precision vs Recall of words).
    Standard metric for SQuAD and QA tasks.
    """
    pred_tokens = predicted_text.lower().split()
    target_tokens = target_text.lower().split()

    if len(pred_tokens) == 0 or len(target_tokens) == 0:
        return int(pred_tokens == target_tokens)

    common_tokens = set(pred_tokens) & set(target_tokens)

    # If no common tokens, F1 is 0
    if len(common_tokens) == 0:
        return 0.0

    prec = len(common_tokens) / len(pred_tokens)
    rec = len(common_tokens) / len(target_tokens)

    return 2 * (prec * rec) / (prec + rec)


def compute_token_f1(pred_ids: list, target_ids: list) -> float:
    """
    Calculates F1 score based on the overlap of Token IDs.
    """
    pred_set = set(pred_ids)
    target_set = set(target_ids)

    # If both are empty, it's a perfect match
    if len(pred_set) == 0 and len(target_set) == 0:
        return 1.0
    # If one is empty but the other isn't, it's a total failure
    elif len(pred_set) == 0 or len(target_set) == 0:
        return 0.0

    common_tokens = pred_set.intersection(target_set)

    if len(common_tokens) == 0:
        return 0.0

    precision = len(common_tokens) / len(pred_set)
    recall = len(common_tokens) / len(target_set)

    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def run_full_validation(
        model,
        validation_dataset,
        tokenizer,
        max_length: int,
        device: str,
        writer,
        global_step: int,
        loss_fn: nn.Module,
        P: float
) -> None:
    meteor = evaluate.load("meteor")

    if hasattr(validation_dataset.dataset, 'change_P'):
        validation_dataset.dataset.change_P(P)

    model.eval()

    total_val_loss = 0.0
    all_f1_scores = []

    expected_texts = []
    predicted_texts = []

    with torch.no_grad():
        for batch in tqdm(validation_dataset, desc="Validating"):
            encoder_input = batch['encoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            decoder_mask = batch['decoder_mask'].to(device)
            label = batch['label'].to(device)

            # Validate loss using teacher forcing --- like during training
            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
            proj_output = model.project(decoder_output)

            loss = loss_fn(proj_output.view(-1, tokenizer.get_vocab_size()), label.view(-1))
            total_val_loss += loss.item()

            # Validate f1 and METEOR on token predictions decoding one token at a time --- real usage scenario
            model_out = _greedy_decode(model, encoder_input, encoder_mask, tokenizer, max_length, device)

            pred_ids = model_out.detach().cpu().numpy().tolist()
            target_ids = tokenizer.encode(batch['target_text'][0]).ids

            f1 = compute_token_f1(pred_ids, target_ids)
            all_f1_scores.append(f1)

            target_text = batch['target_text'][0]
            model_out_text = tokenizer.decode(pred_ids)

            expected_texts.append(target_text)
            predicted_texts.append(model_out_text)


    # Average Loss
    avg_val_loss = total_val_loss / len(validation_dataset)

    # Average F1
    avg_f1 = sum(all_f1_scores) / len(all_f1_scores)

    meteor_result = meteor.compute(predictions=predicted_texts, references=expected_texts)
    meteor_score = meteor_result['meteor']

    print("-" * 80)
    print(f"Validation Step: {global_step}")
    print(f"Average Loss:  {avg_val_loss:.4f}")
    print(f"Token F1:      {avg_f1:.4f}")
    print(f"METEOR Score:  {meteor_score:.4f}")
    print("-" * 80)

    if writer:
        writer.add_scalar('Validation/Loss', avg_val_loss, global_step)
        writer.flush()
        writer.add_scalar('Validation/Token_F1', avg_f1, global_step)
        writer.flush()
        writer.add_scalar('Validation/METEOR', meteor_score, global_step)
        writer.flush()

def run_validation_teacher_forcing():
    pass

def run_validation_visualization():
    pass

def run_test():
    pass

def answer():
    pass

def _greedy_decode(
        model: Transformer,
        encoder_input: torch.Tensor,
        source_mask: torch.Tensor,
        tokenizer: Tokenizer,
        max_length: int,
        device: str
    ) -> torch.Tensor:
    sos_index = tokenizer.token_to_id('<pad>')
    eos_index = tokenizer.token_to_id('</s>')

    encoder_output = model.encode(encoder_input, source_mask)

    decoder_input = torch.empty(1, 1).fill_(sos_index).type_as(encoder_input).to(device)

    while True:

        if decoder_input.size(1) == max_length:
            break

        decoder_input, next_token = _greedy_decode_next_token(model, encoder_output, decoder_input, source_mask, device)

        if next_token == eos_index:
            break

    return decoder_input.squeeze(0)


def _greedy_decode_next_token(
        model: Transformer,
        encoder_output: torch.Tensor,
        decoder_input: torch.Tensor,
        source_mask: torch.Tensor,
        device: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:

    decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)
    out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

    prob = model.project(out[:, -1])
    _, next_token = torch.max(prob, dim = 1)

    decoder_input = torch.cat([decoder_input, torch.empty(1, 1).type_as(decoder_input).fill_(next_token.item()).to(device)], dim = 1)

    return decoder_input, next_token


def predict_sequence(
        model,
        tokenizer,
        source_text: str,
        device: str,
        max_length: int = 100
):
    """
    Predicts the translation/answer for a single source string using the trained model.
    Matches the logic of run_full_validation but for a single raw string.
    """
    model.eval()

    with torch.no_grad():
        sos_id = tokenizer.token_to_id('<pad>')
        eos_id = tokenizer.token_to_id('</s>')
        pad_id = tokenizer.token_to_id('<pad>')

        input_ids = tokenizer.encode(source_text).ids
        input_ids = [sos_id] + input_ids + [eos_id]

        encoder_input = torch.tensor(input_ids).unsqueeze(0).to(device)

        encoder_mask = (encoder_input != pad_id).unsqueeze(0).unsqueeze(0).int().to(device)

        model_out = _greedy_decode(
            model,
            encoder_input,
            encoder_mask,
            tokenizer,
            max_length,
            device
        )


        pred_ids = model_out.detach().cpu().numpy().tolist()

        predicted_text = tokenizer.decode(pred_ids, skip_special_tokens=True)

        return predicted_text, pred_ids


def collect_all_predictions(
        model,
        dataloader,
        tokenizer,
        device: str,
        max_length: int = 100
):
    """
    Iterates through the entire dataset, runs inference, and stores
    Question (Source), True Answer (Target), and Predicted Answer.
    """
    model.eval()

    results = []
    count = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating Predictions"):

            # 1. Prepare Inputs
            encoder_input = batch['encoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)

            # 2. Run Inference (Greedy Decode)
            # This returns the tensor of token IDs
            model_out = _greedy_decode(
                model,
                encoder_input,
                encoder_mask,
                tokenizer,
                max_length,
                device
            )

            pred_ids = model_out.detach().cpu().numpy()
            predicted_text = tokenizer.decode(pred_ids, skip_special_tokens=True)

            true_answer = batch['target_text'][0]

            question = batch['source_text'][0]

            # 6. Store
            results.append({
                "Question": question,
                "True Answer": true_answer,
                "Predicted": predicted_text
            })

            count += 1

    print("-" * 60)
    print(f"Total input pairs processed: {count}")
    print("-" * 60)

    return results


def run_validation(
        model: Transformer,
        validation_dataset: DataLoader,
        tokenizer: Tokenizer,
        max_length: int,
        device: str,
        writer: SummaryWriter,
        global_step: int,
        number_examples: int = 5
    ) -> None:
    validation_dataset.dataset.change_P(0)
    model.eval()

    count = 0

    source_texts = []
    expected = []
    predicted = []


    console_width = 80

    with torch.no_grad():

        for batch in validation_dataset:

            source_text = batch['source_text'][0]
            print('-' * console_width)
            print(f"Source: {source_text}")

            count += 1
            encoder_input = batch['encoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)

            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

            sos_index = tokenizer.token_to_id('<pad>')

            target_ids = tokenizer.encode(batch['target_text'][0]).ids

            next_token = None

            decoder_input_slice = torch.empty(1, 1).fill_(sos_index).type_as(encoder_input).to(device)

            encoder_output = model.encode(encoder_input, encoder_mask)

            next_tokens_predicted = []
            next_tokens_actual = []

            i = 1
            while i <= len(target_ids):

                _, next_token = _greedy_decode_next_token(model, encoder_output, decoder_input_slice, encoder_mask, device)

                next_token = next_token.squeeze(0).detach().cpu().numpy()
                if next_token.ndim == 0:
                    next_token = [next_token.item()]
                elif next_token.ndim == 1:
                    next_token = next_token.tolist()

                next_tokens_predicted.append(tokenizer.decode(next_token))
                next_tokens_actual.append(tokenizer.decode([torch.tensor(target_ids[i - 1]).unsqueeze(0).to(device).squeeze(0).detach().cpu().numpy()]))

                decoder_input_slice = torch.tensor(target_ids[0 : i]).unsqueeze(0).to(device)

                i += 1

            table = [next_tokens_predicted, next_tokens_actual]
            #print(tabulate(table, headers = 'keys', showindex = True, tablefmt = 'grid'))

            model_out = _greedy_decode(model, encoder_input, encoder_mask, tokenizer, max_length, device)

            target_text = batch['target_text'][0]

            model_out_text = tokenizer.decode(model_out.detach().cpu().numpy())

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)

            print(f"Target: {target_text}")
            print(f"Predicted: {model_out_text}")

            if count == number_examples:
                print('-' * console_width)
                break

    if writer:

        metric = torchmetrics.CharErrorRate()
        cer = metric(predicted, expected)
        writer.add_scalar('validation cer', cer, global_step)
        writer.flush()

        metric = torchmetrics.WordErrorRate()
        wer = metric(predicted, expected)
        writer.add_scalar('validation wer', wer, global_step)
        writer.flush()

        metric = torchmetrics.BLEUScore()
        bleu = metric(predicted, expected)
        writer.add_scalar('validation BLEU', bleu, global_step)
        writer.flush()


def _prepare_model(config):
    source_language = config['source_language']
    target_language = config['target_language']

    tokenizer = Tokenizer.from_file(str(config['tokenizer_file'].format(source_language)))

    max_length = config['context_size']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = get_model(config, tokenizer.get_vocab_size()).to(device)
    model_filename = get_latest_weights(config)
    state = torch.load(model_filename)
    model.load_state_dict(state['model_state_dict'])

    pad_token = torch.tensor([tokenizer.token_to_id('[PAD]')], dtype=torch.int64).to(device)

    return {
        'model': model,
        'tokenizer': tokenizer,
        'max_length': max_length,
        'device': device,
        'pad_token': pad_token
    }


def _translate_sentence(
        model: Transformer,
        sentence: str,
        source_tokenizer: Tokenizer,
        target_tokenizer: Tokenizer,
        max_length: int,
        device: str,
        pad_token: torch.Tensor,
        predict_next_tokens: Callable[
            [Transformer, torch.Tensor, torch.Tensor, Tokenizer, Tokenizer, int, str], torch.Tensor] = _greedy_decode
) -> str:
    model.eval()

    with torch.no_grad():
        source_ids = source_tokenizer.encode(sentence).ids
        encoder_input = torch.tensor(source_ids).unsqueeze(0).to(device)

        source_mask = (encoder_input != pad_token).unsqueeze(0).unsqueeze(0).int().to(device)

        model_output = predict_next_tokens(
            model,
            encoder_input,
            source_mask,
            source_tokenizer,
            target_tokenizer,
            max_length,
            device
        )

        translated_sentence = target_tokenizer.decode(model_output.detach().cpu().numpy())

    model.train()

    return translated_sentence


def translate_sentence(sentence: str) -> str:
    config = get_config()

    model_parameters = _prepare_model(config)

    model = model_parameters['model']
    source_tokenizer = model_parameters['source_tokenizer']
    target_tokenizer = model_parameters['target_tokenizer']
    max_length = model_parameters['max_length']
    device = model_parameters['device']
    pad_token = model_parameters['pad_token']

    translation = _translate_sentence(
        model,
        sentence,
        source_tokenizer,
        target_tokenizer,
        max_length,
        device,
        pad_token
    )

    return translation


def translate_sentences(sentences: List[str]) -> List[str]:
    config = get_config()

    model_parameters = _prepare_model(config)

    model = model_parameters['model']
    source_tokenizer = model_parameters['source_tokenizer']
    target_tokenizer = model_parameters['target_tokenizer']
    max_length = model_parameters['max_length']
    device = model_parameters['device']
    pad_token = model_parameters['pad_token']

    translations = []
    for sentence in sentences:
        translations.append((sentence, _translate_sentence(
            model,
            sentence,
            source_tokenizer,
            target_tokenizer,
            max_length,
            device,
            pad_token
        )))

    return translations


def run_test(test_dataset: DataLoader):
    sentences = []

    for batch in test_dataset:
        source_text = batch['source_text'][0]
        sentences.append(source_text)

    translations = translate_sentences(sentences)
    for translation in translations:
        print(f"Original: {translation[0]}")
        print(f"Translated: {translation[1]}")