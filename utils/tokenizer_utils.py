from transformers import BertTokenizer
from utils.logging_utils import log


def tokenize_adversarial_example(input_ids, tokenizer):
    if not isinstance(input_ids, list):
        input_ids = input_ids.squeeze().cpu().tolist()

    # make sure decoded string can be tokenized to the same tokens
    sep_indices = []
    for i, token_id in enumerate(input_ids):
        if token_id == tokenizer.sep_token_id:
            sep_indices.append(i)

    if len(sep_indices) == 1 or tokenizer.sep_token_id == tokenizer.cls_token_id:
        # input is a single text
        decoded = tokenizer.decode(input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        encoded_ids = tokenizer.encode(decoded)
    else:
        # input is a pair of texts
        assert len(sep_indices) == 2, sep_indices
        a_input_ids = input_ids[1:sep_indices[0]]
        b_input_ids = input_ids[sep_indices[0] + 1: sep_indices[1]]
        a_decoded = tokenizer.decode(a_input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        b_decoded = tokenizer.decode(b_input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        encoded_ids = tokenizer.encode(a_decoded, b_decoded)

    return encoded_ids


def valid_tokenization(input_ids, tokenizer: BertTokenizer, verbose=False):
    input_ids = input_ids.squeeze().cpu().tolist()

    if input_ids[0] != tokenizer.cls_token_id:
        input_ids = [tokenizer.cls_token_id] + input_ids
    if input_ids[-1] != tokenizer.sep_token_id:
        input_ids = input_ids + [tokenizer.sep_token_id]

    # make sure decoded string can be tokenized to the same tokens
    encoded_ids = tokenize_adversarial_example(input_ids, tokenizer)
    valid = len(input_ids) == len(encoded_ids) and all(i == j for i, j in zip(input_ids, encoded_ids))
    if verbose and not valid:
        log(f'Inputs: {tokenizer.convert_ids_to_tokens(input_ids)}')
        log(f'Re-encoded: {tokenizer.convert_ids_to_tokens(encoded_ids)}')
    return valid, encoded_ids
