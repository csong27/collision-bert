import argparse
import os
import torch
import tqdm
from transformers import BertTokenizer, glue_processors

from constant import BOS_TOKEN, BERT_LM_MODEL_DIR, PARA_DIR
from models.bert_models import BertForConcatSequenceClassification, BertForLM
from models.scorer import SentenceScorer
from utils.constraints_utils import create_constraints, get_sub_masks, get_inputs_filter_ids, STOPWORDS
from utils.logging_utils import log
from utils.optimization_utils import perturb_logits
from utils.tokenizer_utils import valid_tokenization

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default="0", help='gpu id')
parser.add_argument('--stemp', type=float, default=0.1, help='temperature of softmax')
parser.add_argument('--lr', type=float, default=0.01, help='optimization step size')
parser.add_argument('--max_iter', type=int, default=10, help='maximum iteraiton')
parser.add_argument('--seq_len', type=int, default=32, help='Sequence length')
parser.add_argument('--min_len', type=int, default=5, help='Min sequence length')
parser.add_argument("--beta", default=0.0, type=float, help="Coefficient for language model loss.")
parser.add_argument("--task_name", default='mrpc', type=str)
parser.add_argument("--model_dir", default=PARA_DIR, type=str, help="Path to pre-trained model")
parser.add_argument("--lm_model_dir", default=BERT_LM_MODEL_DIR, type=str, help="Path to pre-trained language model")
parser.add_argument('--perturb_iter', type=int, default=5,  help='PPLM iteration')
parser.add_argument("--kl_scale", default=0.0, type=float, help="KL divergence coefficient")
parser.add_argument("--topk", default=50, type=int, help="Top k sampling for beam search")
parser.add_argument("--num_beams", default=10, type=int, help="Number of beams")
parser.add_argument('--verbose', action='store_true', help='Print every iteration')
parser.add_argument('--nature', action='store_true', help='Nature collision')
parser.add_argument('--regularize', action='store_true', help='Use regularization to decrease perplexity')
parser.add_argument('--fp16', action='store_true', help='fp16')
parser.add_argument("--num_filters", default=500, type=int, help="Number of num_filters words to be filtered")

args = parser.parse_args()


def gen_aggressive_collision(inputs_a, model, tokenizer, device, lm_model=None):
    seq_len = args.seq_len

    word_embedding = model.get_input_embeddings().weight.detach()
    if lm_model is not None:
        lm_word_embedding = lm_model.get_input_embeddings().weight.detach()

    vocab_size = word_embedding.size(0)
    input_ids = tokenizer.encode(inputs_a)
    sub_mask = get_sub_masks(tokenizer, device)
    stopwords_mask = create_constraints(seq_len, tokenizer, device)

    input_mask = torch.zeros(vocab_size, device=device)
    input_ids = torch.tensor(input_ids, device=device).unsqueeze(0)
    # prevent generating the words in the input
    input_mask[input_ids] = -1e9
    batch_input_ids = torch.cat([input_ids] * args.topk, 0)

    def relaxed_to_word_embs(x):
        # convert relaxed inputs to word embedding by softmax attention
        masked_x = x + input_mask + sub_mask
        if args.regularize:
            masked_x += stopwords_mask

        p = torch.softmax(masked_x / args.stemp, -1)
        x = torch.mm(p, word_embedding)
        # add embeddings for period and SEP
        x = torch.cat([x, word_embedding[tokenizer.sep_token_id].unsqueeze(0)])
        return p, x.unsqueeze(0)

    def get_lm_loss(p):
        x = torch.mm(p.detach(), lm_word_embedding).unsqueeze(0)
        return lm_model(inputs_embeds=x, one_hot_labels=p.unsqueeze(0))[0]

    # some constants
    sep_tensor = torch.tensor([tokenizer.sep_token_id] * args.topk, device=device)
    batch_sep_embeds = word_embedding[sep_tensor].unsqueeze(1)
    labels = torch.ones((1,), dtype=torch.long, device=device)

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
            loss = model(input_ids, inputs_embeds=inputs_embeds, labels=labels)[0]
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
                context_embeds = torch.einsum('blv,vh->blh', context, word_embedding)
                context_embeds = torch.cat([context_embeds, batch_sep_embeds], 1)
                clf_logits = model(input_ids=batch_input_ids, inputs_embeds=context_embeds)[0]
                clf_scores = torch.log_softmax(clf_logits, -1)[:, 1].detach().float()
                next_beam_scores.scatter_(0, t_topk_tokens, clf_scores)
                next_clf_scores.append(next_beam_scores.unsqueeze(0))

            next_clf_scores = torch.cat(next_clf_scores, 0)
            next_clf_scores = next_clf_scores + input_mask + sub_mask
            next_scores = next_clf_scores
            if args.regularize:
                next_scores += stopwords_mask[t]

            if output_so_far is None:
                next_scores[1:] = -1e9

            # re-organize to group the beam together
            # (we are keeping top hypothesis across beams)
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

        pad_output_so_far = torch.cat([output_so_far, sep_tensor[:args.num_beams].unsqueeze(1)], 1)
        concat_input_ids = torch.cat([batch_input_ids[:args.num_beams], pad_output_so_far], 1)
        token_type_ids = torch.cat([torch.zeros_like(batch_input_ids[:args.num_beams]),
                                    torch.ones_like(pad_output_so_far)], 1)
        clf_logits = model(input_ids=concat_input_ids, token_type_ids=token_type_ids)[0]
        actual_clf_scores = torch.softmax(clf_logits, -1)[:, 1]
        sorter = torch.argsort(actual_clf_scores, -1, descending=True)
        if args.verbose:
            decoded = [
                f'{actual_clf_scores[i].item() * 100:.2f}%, '
                f'{tokenizer.decode(output_so_far[i].cpu().tolist())}'
                for i in sorter
            ]
            log(f'It={it}, target={inputs_a} | ' + ' | '.join(decoded))

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

        if prev_score >= curr_score:
            break
        prev_score = curr_score

    return best_collision, best_score


def find_filters(query, model, tokenizer, device, k=500):
    words = [w for w in tokenizer.vocab if w.isalpha() and w not in STOPWORDS]
    inputs = tokenizer.batch_encode_plus([[query, w] for w in words],
                                         pad_to_max_length=True)
    all_input_ids = torch.tensor(inputs['input_ids'], device=device)
    all_token_type_ids = torch.tensor(inputs['token_type_ids'], device=device)
    all_attention_masks = torch.tensor(inputs['attention_mask'], device=device)
    n = len(words)
    batch_size = 512
    n_batches = n // batch_size + 1
    all_scores = []
    for i in tqdm.trange(n_batches, desc='Filtering vocab'):
        input_ids = all_input_ids[i * batch_size: (i + 1) * batch_size]
        token_type_ids = all_token_type_ids[i * batch_size: (i + 1) * batch_size]
        attention_masks = all_attention_masks[i * batch_size: (i + 1) * batch_size]
        outputs = model.forward(input_ids, attention_masks, token_type_ids)
        scores = outputs[0][:, 1]
        all_scores.append(scores)

    all_scores = torch.cat(all_scores)
    _, top_indices = torch.topk(all_scores, k)
    filters = set([words[i.item()] for i in top_indices])
    return [w for w in filters if w.isalpha()]


def gen_natural_collision(inputs_a, inputs_b, model, tokenizer, device, lm_model, eval_lm_model=None):
    collition_init = tokenizer.convert_tokens_to_ids([BOS_TOKEN])
    start_idx = 1
    num_beams = args.num_beams
    repetition_penalty = 5.0
    curr_len = len(collition_init)

    filters = find_filters(inputs_a, model, tokenizer, device, k=args.num_filters)
    best_ids = get_inputs_filter_ids(inputs_a, tokenizer)
    best_ids += get_inputs_filter_ids(inputs_b, tokenizer)

    # scores for each sentence in the beam
    beam_scores = torch.zeros((num_beams,), dtype=torch.float, device=device)
    beam_scores[1:] = -1e9

    output_so_far = torch.tensor([collition_init] * num_beams, device=device)
    past = None
    vocab_size = tokenizer.vocab_size
    topk = args.topk
    input_ids = tokenizer.encode(inputs_a)
    input_ids = torch.tensor(input_ids, device=device).unsqueeze(0)
    batch_input_ids = torch.cat([input_ids] * topk, 0)
    sep_tensor = torch.tensor([tokenizer.sep_token_id] * topk, device=device)
    input_mask = torch.zeros(vocab_size, device=device)
    # prevent output num_filters neighbor words
    input_mask[best_ids] = -1e9
    input_mask[tokenizer.convert_tokens_to_ids(['.', '@', '='])] = -1e9
    unk_ids = tokenizer.encode('<unk>', add_special_tokens=False)
    input_mask[unk_ids] = -1e9
    input_mask[tokenizer.convert_tokens_to_ids(filters)] = -1e9

    first_mask = get_sub_masks(tokenizer, device)
    is_first = True
    word_embedding = model.get_input_embeddings().weight.detach()
    batch_sep_embeds = word_embedding[sep_tensor].unsqueeze(1)
    batch_labels = torch.ones((num_beams,), dtype=torch.long, device=device)

    def classifier_loss(p, context):
        context = torch.nn.functional.one_hot(context, len(word_embedding))
        one_hot = torch.cat([context.float(), p.unsqueeze(1)], 1)
        x = torch.einsum('blv,vh->blh', one_hot, word_embedding)
        # add embeddings for SEP
        x = torch.cat([x, batch_sep_embeds[:num_beams]], 1)
        cls_loss = model(batch_input_ids[:num_beams], inputs_embeds=x, labels=batch_labels)[0]
        return cls_loss

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
                verbose=args.verbose
            )

        if repetition_penalty > 1.0:
            lm_model.enforce_repetition_penalty_(next_token_logits, 1, num_beams, output_so_far, repetition_penalty)
        next_token_logits = next_token_logits / args.stemp

        # (batch_size * num_beams, vocab_size)
        next_lm_scores = lm_scores + beam_scores[:, None].expand_as(lm_scores)
        _, topk_tokens = torch.topk(next_token_logits, topk)

        # get target model score here
        next_clf_scores = []
        for i in range(num_beams):
            next_beam_scores = torch.zeros(tokenizer.vocab_size, device=device) - 1e9
            if output_so_far.shape[1] > start_idx:
                curr_beam_topk = output_so_far[i, start_idx:].unsqueeze(0).expand(
                    topk, output_so_far.shape[1] - start_idx)
                # (topk, curr_len + next_token + sep)
                curr_beam_topk = torch.cat([curr_beam_topk, topk_tokens[i].unsqueeze(1), sep_tensor.unsqueeze(1)], 1)
            else:
                curr_beam_topk = torch.cat([topk_tokens[i].unsqueeze(1), sep_tensor.unsqueeze(1)], 1)
            concat_input_ids = torch.cat([batch_input_ids, curr_beam_topk], 1)
            token_type_ids = torch.cat([torch.zeros_like(batch_input_ids), torch.ones_like(curr_beam_topk), ], 1)
            clf_logits = model(input_ids=concat_input_ids, token_type_ids=token_type_ids)[0]
            clf_scores = torch.log_softmax(clf_logits, -1)[:, 1].detach()
            next_beam_scores.scatter_(0, topk_tokens[i], clf_scores.float())
            next_clf_scores.append(next_beam_scores.unsqueeze(0))
        next_clf_scores = torch.cat(next_clf_scores, 0)

        if is_first:
            next_clf_scores += beam_scores[:, None].expand_as(lm_scores)
            next_clf_scores += first_mask
            is_first = False

        next_scores = (1 - args.beta) * next_clf_scores + args.beta * next_lm_scores
        next_scores += input_mask

        # re-organize to group the beam together
        # (we are keeping top hypothesis across beams)
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
        pad_output_so_far = torch.cat([output_so_far[:, start_idx:], sep_tensor[:num_beams].unsqueeze(1)], 1)
        concat_input_ids = torch.cat([batch_input_ids[:num_beams], pad_output_so_far], 1)
        token_type_ids = torch.cat([torch.zeros_like(batch_input_ids[:num_beams]),
                                    torch.ones_like(pad_output_so_far)], 1)
        clf_logits = model(input_ids=concat_input_ids, token_type_ids=token_type_ids)[0]
        actual_clf_scores = torch.softmax(clf_logits, 1)[:, 1]
        sorter = torch.argsort(actual_clf_scores, -1, descending=True)
        if args.verbose:
            decoded = [
                f'{actual_clf_scores[i].item() * 100:.2f}%, '
                f'{tokenizer.decode(output_so_far[i, start_idx:].cpu().tolist())}'
                for i in sorter
            ]
            log(f'Target={inputs_a}, ' + ' | '.join(decoded))

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

    model_dir = os.path.join(args.model_dir, args.task_name.lower())
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    log(f'Loading model from {model_dir}')

    model = BertForConcatSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    eval_lm_model = SentenceScorer(device)
    lm_model = BertForLM.from_pretrained(args.lm_model_dir)
    lm_model.to(device)
    lm_model.eval()
    for param in lm_model.parameters():
        param.requires_grad = False

    if args.fp16:
        from apex import amp
        model, lm_model = amp.initialize([model, lm_model])

    log(f'Loading data from {args.task_name.upper()}')
    data = glue_processors[args.task_name.lower()]().get_dev_examples(model_dir)

    n = 0
    for inputs in data:
        if inputs.label == '1':
            n += 1
            if args.nature:
                collision, score = gen_natural_collision(
                    inputs.text_a, inputs.text_b, model, tokenizer, device,
                    lm_model=lm_model, eval_lm_model=eval_lm_model)
            else:
                collision, score = gen_aggressive_collision(
                    inputs.text_a, model, tokenizer, device, lm_model=lm_model)

            lm_perp = eval_lm_model.perplexity(collision)
            msg = f'Input={inputs.text_a}\n' \
                  f'Ground truth paraphrase={inputs.text_b}\n' \
                  f'Collision={collision}\n' \
                  f'Confidence of being paraphrase={score}\n' \
                  f'LM perp={lm_perp.item()}\n'
            log(msg)


if __name__ == '__main__':
    main()
