import argparse
import json

import numpy as np
import torch
import tqdm
from pattern.text.en import singularize, pluralize

from constant import CONV2AI_PATH, POLY_LM_MODEL_DIR
from models.polyencoder import PretrainedBiEncoder, PretrainedPolyEncoder, PolyEncoderTokenizer, PolyEncoderLM
from models.scorer import SentenceScorer
from utils.constraints_utils import create_poly_constraints, get_poly_sub_masks, get_inputs_filter_ids, STOPWORDS
from utils.logging_utils import log
from utils.optimization_utils import perturb_logits
from utils.tokenizer_utils import valid_tokenization

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default="0", help='gpu id')
parser.add_argument('--stemp', type=float, default=1.0, help='temperature of softmax')
parser.add_argument('--ptemp', type=float, default=1.0, help='temperature of Sinkhorn permutation')
parser.add_argument('--lr', type=float, default=0.001, help='optimization step size')
parser.add_argument('--max_iter', type=int, default=10, help='maximum iteraiton')
parser.add_argument('--seq_len', type=int, default=20, help='Sequence length')
parser.add_argument('--min_len', type=int, default=5, help='Min sequence length')
parser.add_argument("--beta", default=0.0, type=float, help="Coefficient for language model loss.")
parser.add_argument('--verbose', action='store_true', help='Print every iteration')
parser.add_argument('--poly', action='store_true', help='Polyencoder or Biencoder')
parser.add_argument("--lm_model_dir", default=POLY_LM_MODEL_DIR, type=str, help="Path to pre-trained LM")
parser.add_argument('--perturb_iter', type=int, default=5, help='PPLM iteration')
parser.add_argument("--kl_scale", default=0.0, type=float, help="KL divergence coefficient")
parser.add_argument("--topk", default=30, type=int, help="Top k sampling for beam search")
parser.add_argument("--num_beams", default=5, type=int, help="Number of beams")
parser.add_argument('--nature', action='store_true', help='Nature collision')
parser.add_argument('--save', action='store_true', help='Save collision to file')
parser.add_argument("--num_filters", default=1000, type=int, help="Number of num_filters words to be filtered")
parser.add_argument('--regularize', action='store_true', help='Use regularize to decrease perplexity')
parser.add_argument('--fp16', action='store_true', help='fp16')

args = parser.parse_args()


def load_persona_chat():
  with open(CONV2AI_PATH, "r", encoding="utf-8") as f:
    dataset = json.loads(f.read())['valid']

  data_list = []
  for line in dataset:
    persona, utterances = line['personality'], line['utterances']
    for utterance in utterances:
      candidates, history = utterance['candidates'], utterance['history']
      data_list.append((persona, history, candidates))
  return data_list


def gen_aggressive_collision(inputs_a, margin, model, tokenizer, device, lm_model=None):
    seq_len = args.seq_len

    word_embedding = model.get_input_embeddings().weight.detach()
    if lm_model is not None:
        lm_word_embedding = lm_model.get_input_embeddings().weight.detach()

    vocab_size = word_embedding.size(0)
    input_ids = tokenizer.encode(inputs_a)
    sub_mask = get_poly_sub_masks(tokenizer, device)
    stopwords_mask = create_poly_constraints(seq_len, tokenizer, device)
    input_mask = torch.zeros(vocab_size, device=device)
    input_ids = torch.tensor(input_ids, device=device).unsqueeze(0)
    # prevent output num_filters neighbor words
    input_filter_ids = get_inputs_filter_ids(inputs_a, tokenizer)
    input_mask[input_filter_ids] = -1e9
    first_mask = torch.zeros_like(sub_mask)
    first_mask[[tokenizer.dict[w] for w in tokenizer.dict.tok2ind if w.startswith('__')]] = -1e10

    def relaxed_to_word_embs(x):
        # convert relaxed inputs to word embedding by softmax attention
        masked_x = x + input_mask + sub_mask
        if args.regularize:
            masked_x += stopwords_mask

        p = torch.softmax(masked_x / args.stemp, -1)
        x = torch.mm(p, word_embedding)
        # add embeddings for period and SEP
        x = torch.cat([word_embedding[tokenizer.cls_token_id].unsqueeze(0), x,
                       word_embedding[tokenizer.sep_token_id].unsqueeze(0)])
        return p, x.unsqueeze(0)

    def get_lm_loss(p):
        x = torch.mm(p.detach(), lm_word_embedding).unsqueeze(0)
        return lm_model(inputs_embeds=x, one_hot_labels=p.unsqueeze(0))[0]

    # some constants
    cls_tensor = torch.tensor([tokenizer.cls_token_id] * args.topk, device=device)
    sep_tensor = torch.tensor([tokenizer.sep_token_id] * args.topk, device=device)
    batch_cls_emb = word_embedding[cls_tensor].unsqueeze(1)
    batch_sep_emb = word_embedding[sep_tensor].unsqueeze(1)

    best_collision = None
    best_score = -1e9
    prev_score = -1e9

    var_size = (seq_len, vocab_size)
    z_i = torch.zeros(*var_size, requires_grad=True, device=device)
    for it in range(args.max_iter):
        optimizer = torch.optim.Adam([z_i], lr=args.lr)
        for j in range(args.perturb_iter):
            optimizer.zero_grad()
            # relaxation
            p_inputs, inputs_embeds = relaxed_to_word_embs(z_i)
            # forward to BERT with relaxed inputs
            outputs = model(ctxt_input_ids=input_ids, cand_inputs_embeds=inputs_embeds)
            loss = torch.relu(margin - outputs[0].squeeze()).sum()
            if args.beta > 0.:
                lm_loss = get_lm_loss(p_inputs)
                loss = args.beta * lm_loss + (1 - args.beta) * loss
            loss.backward()
            optimizer.step()
            if args.verbose and (j + 1) % 10 == 0:
                log(f'It{it}-{j + 1}, loss={loss.item()}')

        # detach to free GPU memory
        z_i = z_i.detach()

        _, topk_tokens = torch.topk(z_i, args.topk)
        probs_i = torch.softmax(z_i / args.stemp, -1).unsqueeze(0).expand(args.topk, seq_len, vocab_size)

        output_so_far = None
        # beam search left to right
        for t in range(seq_len):
            t_topk_tokens = topk_tokens[t]
            t_topk_onehot = torch.nn.functional.one_hot(t_topk_tokens, vocab_size).float()
            next_clf_scores = []
            for j in range(args.num_beams):
                next_beam_scores = torch.zeros(tokenizer.vocab_size, device=device) - 1e9
                if output_so_far is None:
                    context = probs_i.clone()
                else:
                    output_len = output_so_far.shape[1]
                    beam_topk_output = output_so_far[j].unsqueeze(0).expand(args.topk, output_len)
                    beam_topk_output = torch.nn.functional.one_hot(beam_topk_output, vocab_size)
                    context = torch.cat([beam_topk_output.float(), probs_i[:, output_len:].clone()], 1)
                context[:, t] = t_topk_onehot
                context_emb = torch.einsum('blv,vh->blh', context, word_embedding)

                context_emb = torch.cat([batch_cls_emb, context_emb, batch_sep_emb], 1)
                outputs = model(ctxt_input_ids=input_ids, cand_inputs_embeds=context_emb)[0]
                clf_scores = outputs.squeeze().detach().float()
                next_beam_scores.scatter_(0, t_topk_tokens, clf_scores)
                next_clf_scores.append(next_beam_scores.unsqueeze(0))

            next_clf_scores = torch.cat(next_clf_scores, 0)
            next_scores = next_clf_scores + input_mask + sub_mask

            if args.regularize:
                next_scores += stopwords_mask[t]

            if output_so_far is None:
                next_scores[1:] = -1e9
                next_scores += first_mask

            # re-organize to group the beam together
            # (we are keeping top hypothesis accross beams)
            next_scores = next_scores.view(1, args.num_beams * vocab_size)  # (batch_size, num_beams * vocab_size)
            next_scores, next_tokens = torch.topk(next_scores, args.num_beams, dim=1, largest=True, sorted=True)

            # next batch beam content
            next_sent_beam = []
            for beam_token_rank, (beam_token_id, beam_token_score) in enumerate(zip(next_tokens[0], next_scores[0])):
                # get beam and token IDs
                beam_id = beam_token_id // vocab_size
                token_id = beam_token_id % vocab_size
                next_sent_beam.append((beam_token_score, token_id, beam_id))

            next_batch_beam = next_sent_beam

            # sanity check / prepare next batch
            assert len(next_batch_beam) == args.num_beams
            beam_tokens = torch.tensor([x[1] for x in next_batch_beam], device=device)
            beam_idx = torch.tensor([x[2] for x in next_batch_beam], device=device)

            # re-order batch
            if output_so_far is None:
                output_so_far = beam_tokens.unsqueeze(1)
            else:
                output_so_far = output_so_far[beam_idx, :]
                output_so_far = torch.cat([output_so_far, beam_tokens.unsqueeze(1)], dim=-1)

        pad_output_so_far = torch.cat([cls_tensor[:args.num_beams].unsqueeze(1),
                                       output_so_far,
                                       sep_tensor[:args.num_beams].unsqueeze(1)], 1)
        actual_clf_scores = model(ctxt_input_ids=input_ids, cand_input_ids=pad_output_so_far)[0]
        actual_clf_scores = actual_clf_scores.squeeze()
        sorter = torch.argsort(actual_clf_scores, -1, descending=True)
        if args.verbose:
            decoded = [
                f'{actual_clf_scores[i].item():.2f}, '
                f'{tokenizer.decode(output_so_far[i].cpu().tolist())}'
                for i in sorter
            ]
            log(f'It={it}, margin={margin}, target={inputs_a} | ' + ' | '.join(decoded))

        valid_idx = sorter[0]
        valid = False
        for idx in sorter:
            valid, _ = valid_tokenization(output_so_far[idx], tokenizer)
            if valid:
                valid_idx = idx
                break

        # re-initialize z_i
        curr_best = output_so_far[valid_idx]
        next_z_i = torch.nn.functional.one_hot(curr_best, vocab_size).float()
        eps = 0.1
        next_z_i = (next_z_i * (1 - eps)) + (1 - next_z_i) * eps / (vocab_size - 1)
        z_i = torch.nn.Parameter(torch.log(next_z_i), True)

        curr_score = actual_clf_scores[valid_idx].item()
        if valid and curr_score > best_score:
            best_score = curr_score
            best_collision = tokenizer.decode(curr_best.cpu().tolist())

        if curr_score == prev_score:
            break
        prev_score = curr_score

    return best_collision, best_score


def find_filters(query, model, tokenizer, device, k=500):
    words = [w for w in tokenizer.dict.tok2ind if w.isalpha() and w not in STOPWORDS]
    inputs_a = torch.tensor(tokenizer.encode(query), device=device).unsqueeze(0)
    inputs_b = [tokenizer.encode(w) for w in words]
    inputs_b = torch.tensor(inputs_b, device=device)
    n = len(words)
    batch_size = 1024
    n_batches = n // batch_size + 1
    all_scores = []
    for i in tqdm.trange(n_batches, desc='Filtering vocab'):
        cand_input_ids = inputs_b[i * batch_size: (i + 1) * batch_size]
        scores = model(ctxt_input_ids=inputs_a, cand_input_ids=cand_input_ids)[0].squeeze()
        all_scores.append(scores)
    all_scores = torch.cat(all_scores)
    _, top_indices = torch.topk(all_scores, k)
    filters = set([words[i.item()] for i in top_indices])
    return [w for w in filters if w.isalpha()]


def add_single_plural(text, tokenizer):
    tokens = tokenizer.tokenize(text)
    vocab = tokenizer.get_vocab()
    contains = []
    for word in vocab:
        if word.isalpha() and len(word) > 2:
            for t in tokens:
                if len(t) > 2 and word != t and (word.startswith(t) or t.startswith(word)):
                    contains.append(word)

    for t in tokens[:]:
        if not t.isalpha():
            continue
        sig_t = singularize(t)
        plu_t = pluralize(t)
        if sig_t != t and sig_t in vocab:
            tokens.append(sig_t)
        if plu_t != t and plu_t in vocab:
            tokens.append(plu_t)
    return [w for w in tokens + contains if w not in STOPWORDS]


def gen_natural_collision(inputs_a, inputs_b, model, tokenizer, device, lm_model, eval_lm_model, margin):
    filters = find_filters(inputs_a, model, tokenizer, device, args.num_filters)
    num_filters_ids = tokenizer.convert_tokens_to_ids(filters)

    newline_token_id = tokenizer.dict['__newln__']

    sub_mask = torch.zeros(tokenizer.vocab_size, device=device)
    sub_mask[newline_token_id] = -1e9
    filter_ids = [tokenizer.dict[w] for w in tokenizer.dict.tok2ind if not w.isalnum()]
    first_mask = torch.zeros_like(sub_mask)
    first_mask[filter_ids] = -1e9

    collition_init = [tokenizer.dict['__start__']]
    start_idx = 1
    num_beams = args.num_beams
    repetition_penalty = 5.0
    curr_len = len(collition_init)

    # scores for each sentence in the beam
    beam_scores = torch.zeros((num_beams,), dtype=torch.float, device=device)
    beam_scores[1:] = -1e9

    output_so_far = torch.tensor([collition_init] * num_beams, device=device)
    past = None
    vocab_size = tokenizer.vocab_size
    topk = args.topk
    input_ids = tokenizer.encode(inputs_a)
    input_mask = torch.zeros(vocab_size, device=device)
    # prevent output num_filters neighbor words
    input_filter_ids = get_inputs_filter_ids(inputs_a, tokenizer) + get_inputs_filter_ids(inputs_b, tokenizer)
    remove_tokens = add_single_plural(inputs_a, tokenizer)
    remove_ids = tokenizer.convert_tokens_to_ids(remove_tokens)
    input_mask[remove_ids] = -1e10
    input_mask[input_filter_ids] = -1e9
    input_mask[tokenizer.dict['.']] = -1e9
    input_mask[tokenizer.dict['=']] = -1e9
    input_mask[tokenizer.dict['@']] = -1e9
    input_mask[tokenizer.dict['@@.']] = -1e9
    input_mask[tokenizer.dict['__null__']] = -1e9
    input_mask[num_filters_ids] = -1e9
    unk_ids = tokenizer.encode('<unk>', add_special_tokens=False)
    input_mask[unk_ids] = -1e9
    input_mask[[tokenizer.dict[w] for w in tokenizer.dict.tok2ind if w.startswith('__')]] = -1e10

    input_ids = torch.tensor(input_ids, device=device).unsqueeze(0)
    sep_tensor = torch.tensor([tokenizer.sep_token_id] * topk, device=device)
    cls_tensor = torch.tensor([tokenizer.cls_token_id] * topk, device=device)

    is_first = True
    word_embedding = model.get_input_embeddings().weight.detach()
    batch_sep_emb = word_embedding[sep_tensor].unsqueeze(1)
    batch_cls_emb = word_embedding[cls_tensor].unsqueeze(1)

    def classifier_loss(p, context):
        context = torch.nn.functional.one_hot(context, len(word_embedding))
        one_hot = torch.cat([context.float(), p.unsqueeze(1)], 1)
        x = torch.einsum('blv,vh->blh', one_hot, word_embedding)
        # add embeddings for SEP
        x = torch.cat([batch_cls_emb[:num_beams], x, batch_sep_emb[:num_beams]], 1)
        scores = model(ctxt_input_ids=input_ids, cand_inputs_embeds=x)[0]
        loss = (margin - scores.squeeze()).mean()
        return loss

    best_score = -1e9
    best_collision = None

    while curr_len < args.seq_len:
        model_inputs = lm_model.prepare_inputs_for_generation(output_so_far, past=past)
        outputs = lm_model(**model_inputs)
        present = outputs[1]
        # (batch_size * num_beams, vocab_size)
        next_token_logits = outputs[0][:, -1, :]
        lm_scores = torch.log_softmax(next_token_logits, dim=-1)

        if args.perturb_iter > 0:
            # perturb internal states of LM
            def target_model_wrapper(p):
                return classifier_loss(p, output_so_far.detach()[:, start_idx:])

            next_token_logits = perturb_logits(
                next_token_logits,
                args.lr,
                target_model_wrapper,
                num_iterations=args.perturb_iter,
                kl_scale=args.kl_scale,
                temperature=args.stemp,
                device=device,
                verbose=args.verbose,
                logit_mask=input_mask,
            )

        if repetition_penalty > 1.0:
            lm_model.enforce_repetition_penalty_(next_token_logits, 1, num_beams, output_so_far, repetition_penalty)
        next_token_logits = next_token_logits / args.stemp

        # (batch_size * num_beams, vocab_size)
        next_lm_scores = lm_scores + beam_scores[:, None].expand_as(lm_scores) + sub_mask
        _, topk_tokens = torch.topk(next_token_logits, topk)

        # get target model score here
        next_clf_scores = []
        for i in range(num_beams):
            next_beam_scores = torch.zeros(tokenizer.vocab_size, device=device) - 1e9
            if output_so_far.shape[1] > start_idx:
                curr_beam_topk = output_so_far[i, start_idx:].unsqueeze(0).expand(
                    topk, output_so_far.shape[1] - start_idx)
                # (topk, curr_len + next_token + sep)
                curr_beam_topk = torch.cat([cls_tensor.unsqueeze(1), curr_beam_topk,
                                            topk_tokens[i].unsqueeze(1),
                                            sep_tensor.unsqueeze(1)], 1)
            else:
                curr_beam_topk = torch.cat([cls_tensor.unsqueeze(1),
                                            topk_tokens[i].unsqueeze(1),
                                            sep_tensor.unsqueeze(1)], 1)
            clf_scores = model(ctxt_input_ids=input_ids, cand_input_ids=curr_beam_topk)[0].squeeze()
            next_beam_scores.scatter_(0, topk_tokens[i], clf_scores.float())
            next_clf_scores.append(next_beam_scores.unsqueeze(0))
        next_clf_scores = torch.cat(next_clf_scores, 0)

        if is_first:
            next_clf_scores += beam_scores[:, None].expand_as(lm_scores)
            next_clf_scores += first_mask
            is_first = False

        next_scores = (1 - args.beta) * next_clf_scores + args.beta * next_lm_scores
        next_scores += input_mask

        # re-organize to group the beam together (we are keeping top hypothesis accross beams)
        next_scores = next_scores.view(num_beams * vocab_size)
        next_lm_scores = next_lm_scores.view(num_beams * vocab_size)
        next_scores, next_tokens = torch.topk(next_scores, num_beams, largest=True, sorted=True)
        next_lm_scores = next_lm_scores[next_tokens]
        # next batch beam content
        next_sent_beam = []
        for beam_token_rank, (beam_token_id, beam_token_score) in enumerate(zip(next_tokens, next_lm_scores)):
            # get beam and token IDs
            beam_id = beam_token_id // vocab_size
            token_id = beam_token_id % vocab_size
            next_sent_beam.append((beam_token_score, token_id, beam_id))

        next_batch_beam = next_sent_beam

        # sanity check / prepare next batch
        assert len(next_batch_beam) == num_beams
        beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
        beam_tokens = output_so_far.new([x[1] for x in next_batch_beam])
        beam_idx = output_so_far.new([x[2] for x in next_batch_beam])

        # re-order batch
        output_so_far = output_so_far[beam_idx, :]
        output_so_far = torch.cat([output_so_far, beam_tokens.unsqueeze(1)], dim=-1)

        # sanity check
        pad_output_so_far = torch.cat([cls_tensor[: num_beams].unsqueeze(1),
                                       output_so_far[:, start_idx:],
                                       sep_tensor[:num_beams].unsqueeze(1)], 1)
        actual_clf_scores = model(ctxt_input_ids=input_ids, cand_input_ids=pad_output_so_far)[0].squeeze()

        sorter = torch.argsort(actual_clf_scores, -1, descending=True)
        if args.verbose:
            decoded = [
                f'{actual_clf_scores[i].item():.2f}, '
                f'{tokenizer.decode(output_so_far[i, start_idx:].cpu().tolist())}'
                for i in sorter
            ]
            log(f'Margin={margin}, target={inputs_a} | ' + ' | '.join(decoded))

        if curr_len > args.min_len:
            valid_idx = sorter[0]
            valid = False
            for idx in sorter:
                valid, _ = valid_tokenization(output_so_far[idx, start_idx:], tokenizer)
                if valid:
                    valid_idx = idx
                    break

            curr_score = actual_clf_scores[valid_idx].item()
            curr_collision = tokenizer.decode(output_so_far[valid_idx, start_idx:].cpu().tolist())
            if valid and curr_score > best_score:
                best_score = curr_score
                best_collision = curr_collision

            if args.verbose:
                lm_perp = eval_lm_model.perplexity(curr_collision)
                log(f'LM perp={lm_perp.item()}')

        # re-order internal states
        past = lm_model._reorder_cache(present, beam_idx)
        # update current length
        curr_len = curr_len + 1

    return best_collision, best_score


def main():
    device = torch.device(f'cuda:{args.gpu}')
    data_list = load_persona_chat()
    tokenizer = PolyEncoderTokenizer.from_pretrained()
    if args.poly:
        model = PretrainedPolyEncoder.from_pretrained()
    else:
        model = PretrainedBiEncoder.from_pretrained()

    history_size = model.opt['history_size']
    text_truncate = model.opt['text_truncate']
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    log(f'Loading LM model from {args.lm_model_dir}')
    lm_model = PolyEncoderLM.from_pretrained(checkpoint=args.lm_model_dir)
    lm_model.to(device)
    lm_model.eval()
    for param in lm_model.parameters():
        param.requires_grad = False
    eval_lm_model = SentenceScorer(device)

    if args.fp16:
        from apex import amp
        model, lm_model = amp.initialize([model, lm_model])

    for data in data_list:
        query = '\n'.join(data[0] + data[1][-history_size:])
        candidates = data[2]
        truth = candidates[-1]
        query_ids = torch.tensor(tokenizer.encode(query, max_length=text_truncate), device=device).unsqueeze(0)
        candidates = torch.tensor(tokenizer.batch_encode_plus(
            candidates, pad_to_max_length=True)['input_ids'], device=device)
        output = model.forward(ctxt_input_ids=query_ids, cand_input_ids=candidates)
        scores = output[0].squeeze()
        if args.nature:
            collision, score = gen_natural_collision(
                query, truth, model, tokenizer, device, lm_model, eval_lm_model, scores.max())
        else:
            collision, score = gen_aggressive_collision(query, scores.max(), model, tokenizer, device, lm_model)

        # get the rank of collision
        scores = scores.cpu().tolist()
        scores = np.asarray([score] + scores)
        n = len(scores)
        ranks = np.empty(n)
        ranks[np.argsort(-scores)] = np.arange(n) + 1

        lm_perp = eval_lm_model.perplexity(collision)
        msg = f'Input={query}\n' \
              f'Ground truth response={truth}\n' \
              f'Collision={collision}\n' \
              f'Collision similarity core={score}\n' \
              f'Rank={ranks[0]}\n' \
              f'LM perp={lm_perp.item()}\n'
        log(msg)


if __name__ == '__main__':
    main()
