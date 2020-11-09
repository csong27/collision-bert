import torch
from nltk.corpus import stopwords
from transformers import BertTokenizer, GPT2Tokenizer

COMMON_WORDS = ['the', 'of', 'and', 'a', 'to', 'in', 'is', 'you', 'that', 'it']
STOPWORDS = set(stopwords.words('english'))


def get_inputs_filter_ids(inputs, tokenizer):
    tokens = [w for w in tokenizer.tokenize(inputs) if w.isalpha() and w not in STOPWORDS]
    return tokenizer.convert_tokens_to_ids(tokens)


def get_sub_masks(tokenizer, device, prob=False):
    # masking for all subwords in the vocabulary
    vocab = tokenizer.get_vocab()

    def is_special_token(w):
        if isinstance(tokenizer, BertTokenizer) and w.startswith('##'):
            return True
        if isinstance(tokenizer, GPT2Tokenizer) and not w.startswith('Ġ'):
            return True
        if w[0] == '[' and w[-1] == ']':
            return True
        if w[0] == '<' and w[-1] == '>':
            return True
        if w in ['=', '@', 'Ġ=', 'Ġ@'] and w in vocab:
            return True
        return False

    filter_ids = [vocab[w] for w in vocab if is_special_token(w)]
    if prob:
        prob_mask = torch.ones(tokenizer.vocab_size, device=device)
        prob_mask[filter_ids] = 0.
    else:
        prob_mask = torch.zeros(tokenizer.vocab_size, device=device)
        prob_mask[filter_ids] = -1e9
    return prob_mask


def get_poly_sub_masks(tokenizer, device, prob=False):
    filter_ids = [tokenizer.dict[w] for w in tokenizer.dict.tok2ind
                  if not w.isalnum()]
    if prob:
        prob_mask = torch.ones(tokenizer.vocab_size, device=device)
        prob_mask[filter_ids] = 0.
    else:
        prob_mask = torch.zeros(tokenizer.vocab_size, device=device)
        prob_mask[filter_ids] = -1e9
    return prob_mask


def create_constraints(seq_len, tokenizer, device, prob=False):
    stopword_ids = [tokenizer.vocab[w] for w in COMMON_WORDS[:5] if w in tokenizer.vocab]
    if prob:
        masks = torch.zeros(seq_len, tokenizer.vocab_size, device=device)
    else:
        masks = torch.zeros(seq_len, tokenizer.vocab_size, device=device) - 1e9

    for t in range(seq_len):
        if t >= seq_len // 2:
            masks[t, stopword_ids] = 1.0 if prob else 0.0
        else:
            masks[t] = 1.0 if prob else 0.
    return masks


def create_poly_constraints(seq_len, tokenizer, device, prob=False):
    stopword_ids = [tokenizer.dict[w] for w in COMMON_WORDS[:5] if w in tokenizer.dict.tok2ind]
    if prob:
        masks = torch.zeros(seq_len, tokenizer.vocab_size, device=device)
    else:
        masks = torch.zeros(seq_len, tokenizer.vocab_size, device=device) - 1e9

    for t in range(seq_len):
        if t >= seq_len // 3:
            masks[t, stopword_ids] = 1.0 if prob else 0.0
        else:
            masks[t] = 1.0 if prob else 0.
    return masks
