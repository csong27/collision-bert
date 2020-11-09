import argparse
import bisect
import os

import apex.amp as amp
import torch
from transformers import BertTokenizer

from constant import BOS_TOKEN, PRESUMM_DATA_DIR, PRESUMM_MODEL_PATH, BERT_LM_MODEL_DIR
from models.bert_models import BertForLM
from models.presumm.config import get_config
from models.presumm.model_builder import ExtSummarizer
from models.scorer import SentenceScorer
from utils.constraints_utils import create_constraints, get_sub_masks, STOPWORDS
from utils.logging_utils import log
from utils.optimization_utils import perturb_logits
from utils.tokenizer_utils import valid_tokenization

CONFIG = get_config()

parser = argparse.ArgumentParser(description='squad')
parser.add_argument('--gpu', type=str, default="0", help='gpu id')
parser.add_argument('--stemp', type=float, default=1.0, help='temperature of softmax')
parser.add_argument('--lr', type=float, default=0.001, help='optimization step size')
parser.add_argument('--max_iter', type=int, default=10, help='maximum iteraiton')
parser.add_argument('--seq_len', type=int, default=20, help='Sequence length')
parser.add_argument('--min_len', type=int, default=5, help='Min sequence length')
parser.add_argument("--beta", default=0.0, type=float, help="Coefficient for language model loss.")
parser.add_argument('--perturb_iter', type=int, default=5, help='PPLM iteration')
parser.add_argument("--kl_scale", default=0.0, type=float, help="KL divergence coefficient")
parser.add_argument("--topk", default=30, type=int, help="Top k sampling for beam search")
parser.add_argument("--num_beams", default=5, type=int, help="Number of beams")
parser.add_argument('--save', action='store_true', help='Save collision to file')
parser.add_argument("--lm_model_dir", default=BERT_LM_MODEL_DIR, type=str, help="Path to pre-trained LM")
parser.add_argument("--insert_pos", default=0.1, type=float, help="Relative position to insert collision into document")
parser.add_argument('--verbose', action='store_true', help='Print every iteration')
parser.add_argument('--nature', action='store_true', help='Nature collision')
parser.add_argument('--fp16', action='store_true', help='Mix precision')
parser.add_argument('--regularize', action='store_true', help='Use regularize to decrease perplexity')


def preprocess(ex):
    src = ex['src']
    src_sent_labels = ex['src_sent_labels']
    segs = ex['segs']
    if not CONFIG.use_interval:
        segs = [0] * len(segs)
    clss = ex['clss']
    src_txt = ex['src_txt']
    tgt_txt = ex['tgt_txt']

    end_id = [src[-1]]
    src = src[:-1][:CONFIG.max_pos - 1] + end_id
    segs = segs[:CONFIG.max_pos]
    max_sent_id = bisect.bisect_left(clss, CONFIG.max_pos)
    src_sent_labels = src_sent_labels[:max_sent_id]
    clss = clss[:max_sent_id]
    return src, segs, clss, src_sent_labels, src_txt, tgt_txt


def load_ext_sum_data(split='valid'):
    data = []
    for filename in sorted(os.listdir(PRESUMM_DATA_DIR)):
        if split in filename:
            filename = os.path.join(PRESUMM_DATA_DIR, filename)
            data += torch.load(filename)
    return data


def get_input_constant(label, seq_len, src_ids, src_embeds, segs, clss, device):
    offset = clss[label]
    type_token_ids = segs[:offset] + [1 - segs[:offset][-1]] * (seq_len + 2) + [1 - s for s in segs[offset:]]
    new_clss = clss[:label + 1] + [cls + (seq_len + 2) for cls in clss[label:]]
    prefix_embeds = src_embeds[:offset]
    batch_prefix_ids = torch.cat([src_ids[:offset].unsqueeze(0)] * args.topk, 0)
    batch_prefix_emb = torch.cat([prefix_embeds.unsqueeze(0)] * args.topk, 0)
    src_embeds = src_embeds[offset:]
    batch_src_ids = torch.cat([src_ids[offset:].unsqueeze(0)] * args.topk, 0)
    batch_src_emb = torch.cat([src_embeds.unsqueeze(0)] * args.topk, 0)

    type_token_ids = torch.tensor(type_token_ids, device=device).unsqueeze(0)
    new_clss = torch.tensor(new_clss, device=device).unsqueeze(0)
    mask_cls = torch.ones_like(new_clss, device=device)
    batch_segs = torch.cat([type_token_ids] * args.topk, 0)
    batch_new_clss = torch.cat([new_clss] * args.topk, 0)
    return batch_prefix_ids, batch_prefix_emb, batch_src_ids, batch_src_emb, mask_cls, batch_segs, batch_new_clss


def gen_aggressive_collision(ex, model, tokenizer, device, lm_model=None):
    src, segs, clss, src_sent_labels, src_txt, tgt_txt = ex

    word_embedding = model.bert.model.get_input_embeddings().weight.detach()
    if lm_model is not None:
        lm_word_embedding = lm_model.get_input_embeddings().weight.detach()

    vocab_size = word_embedding.size(0)
    src_ids = torch.tensor(src, device=device)
    src_embeds = word_embedding[src_ids]

    sub_mask = get_sub_masks(tokenizer, device)
    input_mask = torch.zeros(vocab_size, device=device)
    src_tokens = [w for w in tokenizer.convert_ids_to_tokens(src) if w.isalpha() and w not in STOPWORDS]
    input_mask[tokenizer.convert_tokens_to_ids(src_tokens)] = -1e9
    seq_len = args.seq_len
    stopwords_mask = create_constraints(seq_len, tokenizer, device)

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
    sep_tensor = torch.tensor([tokenizer.sep_token_id] * args.topk, device=device)
    batch_sep_emb = word_embedding[sep_tensor].unsqueeze(1)
    cls_tensor = torch.tensor([tokenizer.cls_token_id] * args.topk, device=device)
    batch_cls_emb = word_embedding[cls_tensor].unsqueeze(1)

    label = int(len(clss) * args.insert_pos)
    labels = torch.tensor([label], device=device)
    batch_prefix_ids, batch_prefix_emb, batch_src_ids, batch_src_emb, mask_cls, batch_segs, batch_new_clss = \
        get_input_constant(label, seq_len, src_ids, src_embeds, segs, clss, device)
    prefix_embeds = batch_prefix_emb[0]
    src_embeds = batch_src_emb[0]
    type_token_ids = batch_segs[0]
    new_clss = batch_new_clss[0]

    loss_fn = torch.nn.CrossEntropyLoss()

    best_collision = None
    best_score = -1e9
    best_rank = -1
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
            inputs_embeds = torch.cat([prefix_embeds.unsqueeze(0), inputs_embeds, src_embeds.unsqueeze(0)], 1)
            scores = model(None, type_token_ids, new_clss, None, mask_cls, inputs_embeds, output_logits=True)
            loss = loss_fn(scores, labels)
            scores = scores.squeeze()
            loss += torch.max(scores) - scores[label]
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
                inputs_emb = torch.cat([batch_prefix_emb, context_emb, batch_src_emb], 1)
                scores = model(None, batch_segs, batch_new_clss, None, mask_cls, inputs_emb, output_logits=True)
                clf_scores = scores[:, label].detach().float()
                next_beam_scores.scatter_(0, t_topk_tokens, clf_scores)
                next_clf_scores.append(next_beam_scores.unsqueeze(0))

            next_clf_scores = torch.cat(next_clf_scores, 0)
            next_scores = next_clf_scores + input_mask + sub_mask
            if args.regularize:
                next_scores += stopwords_mask[t]

            if output_so_far is None:
                next_scores[1:] = -1e9

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
        concat_input_ids = torch.cat([batch_prefix_ids[:args.num_beams],
                                      pad_output_so_far,
                                      batch_src_ids[:args.num_beams]], 1)
        actual_scores = model.forward(concat_input_ids,
                                      batch_segs[: args.num_beams],
                                      batch_new_clss[:args.num_beams],
                                      None, mask_cls, None).squeeze()
        actual_clf_scores = actual_scores[:, label].detach()
        top_scores, top_labels = torch.topk(actual_scores, actual_scores.shape[-1])
        sorter = torch.argsort(actual_clf_scores, -1, descending=True)
        if args.verbose:
            decoded = [
                f'{actual_clf_scores[i].item():.4f}, '
                f'{tokenizer.decode(output_so_far[i].cpu().tolist())}'
                for i in sorter
            ]
            log(f'It={it}, margin={top_scores[:, 2].max().item()} | ' + ' | '.join(decoded))

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
        curr_collision = tokenizer.decode(curr_best.cpu().tolist())
        curr_rank = (top_labels[valid_idx] == label).nonzero().squeeze().item()
        if valid and curr_score > best_score:
            best_score = curr_score
            best_collision = curr_collision
            best_rank = curr_rank

        if prev_score == curr_score:
            break
        prev_score = curr_score

    return best_collision, best_score, best_rank


def gen_natural_collision(ex, model, tokenizer, device, lm_model, eval_lm_model=None):
    src, segs, clss, src_sent_labels, src_txt, tgt_txt = ex
    word_embedding = model.bert.model.get_input_embeddings().weight.detach()
    collition_init = tokenizer.convert_tokens_to_ids([BOS_TOKEN])

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
    src_ids = torch.tensor(src, device=device)
    src_embeds = word_embedding[src_ids]

    sub_mask = get_sub_masks(tokenizer, device)
    filter_ids = [tokenizer.vocab[w] for w in tokenizer.vocab if not w.isalnum()]
    first_mask = torch.zeros_like(sub_mask)
    first_mask[filter_ids] = -1e9
    input_mask = torch.zeros(vocab_size, device=device)
    src_tokens = [w for w in tokenizer.convert_ids_to_tokens(src) if w.isalpha() and w not in STOPWORDS]
    input_mask[tokenizer.convert_tokens_to_ids(src_tokens)] = -1e9
    input_mask[tokenizer.convert_tokens_to_ids(['.', '@', '='])] = -1e9
    unk_ids = tokenizer.encode('<unk>', add_special_tokens=False)
    input_mask[unk_ids] = -1e9

    sep_tensor = torch.tensor([tokenizer.sep_token_id] * topk, device=device)
    cls_tensor = torch.tensor([tokenizer.cls_token_id] * topk, device=device)

    is_first = True
    batch_sep_emb = word_embedding[sep_tensor].unsqueeze(1)
    batch_cls_emb = word_embedding[cls_tensor].unsqueeze(1)
    label = int(len(clss) * args.insert_pos)
    labels = torch.tensor([label] * num_beams, device=device)
    loss_fn = torch.nn.CrossEntropyLoss()

    def classifier_loss(p, context, pre_emb, src_emb, type_token_ids, new_clss, mask):
        context = torch.nn.functional.one_hot(context, len(word_embedding))
        one_hot = torch.cat([context.float(), p.unsqueeze(1)], 1)
        x = torch.einsum('blv,vh->blh', one_hot, word_embedding)
        # add embeddings for SEP
        x = torch.cat([batch_cls_emb[:num_beams], x, batch_sep_emb[:num_beams]], 1)
        inputs_embeds = torch.cat([pre_emb, x, src_emb], 1)
        scores = model(None, type_token_ids, new_clss, None, mask, inputs_embeds, output_logits=True)
        loss = loss_fn(scores, labels)
        loss += torch.mean(torch.max(scores, 1)[0] - scores[:, label])
        return loss

    best_collision = None
    best_score = -1e9
    best_rank = -1

    while curr_len < args.seq_len:
        seq_len = curr_len - start_idx + 1
        batch_prefix_ids, batch_prefix_emb, batch_src_ids, batch_src_emb, mask_cls, batch_segs, batch_new_clss = \
            get_input_constant(label, seq_len, src_ids, src_embeds, segs, clss, device)
        model_inputs = lm_model.prepare_inputs_for_generation(output_so_far, past=past)
        outputs = lm_model(**model_inputs)
        present = outputs[1]
        # (batch_size * num_beams, vocab_size)
        next_token_logits = outputs[0][:, -1, :]
        lm_scores = torch.log_softmax(next_token_logits, dim=-1)
        next_lm_scores = lm_scores + beam_scores[:, None].expand_as(lm_scores)

        if args.perturb_iter > 0:
            # perturb internal states of LM
            def target_model_wrapper(p):
                return classifier_loss(p, output_so_far.detach()[:, start_idx:],
                                       batch_prefix_emb[:num_beams],
                                       batch_src_emb[:num_beams],
                                       batch_segs[:num_beams],
                                       batch_new_clss[:num_beams], mask_cls)

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
        _, topk_tokens = torch.topk(next_token_logits, topk)

        # get target model score here
        next_clf_scores = []
        for i in range(num_beams):
            next_beam_scores = torch.zeros(tokenizer.vocab_size, device=device) - 1e9
            if output_so_far.shape[1] > start_idx:
                curr_beam_topk = output_so_far[i, start_idx:].unsqueeze(0).expand(
                    topk, output_so_far.shape[1] - start_idx)
                # (topk, curr_len + next_token + sep)
                curr_beam_topk = torch.cat([cls_tensor.unsqueeze(1),
                                            curr_beam_topk,
                                            topk_tokens[i].unsqueeze(1),
                                            sep_tensor.unsqueeze(1)], 1)
            else:
                curr_beam_topk = torch.cat([cls_tensor.unsqueeze(1),
                                            topk_tokens[i].unsqueeze(1),
                                            sep_tensor.unsqueeze(1)], 1)
            concat_input_ids = torch.cat([batch_prefix_ids, curr_beam_topk, batch_src_ids], 1)
            scores = model(concat_input_ids, batch_segs, batch_new_clss, None, mask_cls, None)
            clf_scores = torch.log_softmax(scores, -1)[:, label].detach()
            next_beam_scores.scatter_(0, topk_tokens[i], clf_scores)
            next_clf_scores.append(next_beam_scores.unsqueeze(0))
        next_clf_scores = torch.cat(next_clf_scores, 0)

        if is_first:
            next_clf_scores += beam_scores[:, None].expand_as(lm_scores)
            next_clf_scores += first_mask
            is_first = False

        next_scores = (1 - args.beta) * next_clf_scores + args.beta * next_lm_scores
        next_scores += input_mask

        # re-organize to group the beam together
        # (we are keeping top hypothesis accross beams)
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
        pad_output_so_far = torch.cat([cls_tensor[:num_beams].unsqueeze(1),
                                       output_so_far[:, start_idx:],
                                       sep_tensor[:num_beams].unsqueeze(1)], 1)
        concat_input_ids = torch.cat([batch_prefix_ids[:num_beams],
                                      pad_output_so_far,
                                      batch_src_ids[:num_beams]], 1)
        actual_scores = model.forward(concat_input_ids,
                                      batch_segs[:num_beams],
                                      batch_new_clss[:num_beams],
                                      None, mask_cls, None)
        top_scores, top_labels = torch.topk(actual_scores, actual_scores.shape[-1])
        actual_clf_scores = actual_scores[:, label].detach()
        sorter = torch.argsort(actual_clf_scores, -1, descending=True)
        if args.verbose:
            decoded = [
                f'{actual_clf_scores[i].item():.4f}, '
                f'{tokenizer.decode(output_so_far[i, start_idx:].cpu().tolist())}'
                for i in sorter
            ]
            log(f'Margin={top_scores[:, 2].max().item()} | ' + ' | '.join(decoded))

        # re-order internal states
        past = lm_model._reorder_cache(present, beam_idx)
        # update current length
        curr_len = curr_len + 1

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
            curr_rank = (top_labels[valid_idx] == label).nonzero().squeeze().item()
            if valid and curr_score > best_score:
                best_score = curr_score
                best_collision = curr_collision
                best_rank = curr_rank
            lm_perp = eval_lm_model.perplexity(curr_collision)
            log(f'LM perp={lm_perp.item()}')

    return best_collision, best_score, best_rank


def main():
    device = torch.device(f'cuda:{args.gpu}')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = ExtSummarizer(CONFIG, torch.load(PRESUMM_MODEL_PATH))
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    if args.fp16:
        model = amp.initialize(model)

    eval_lm_model = SentenceScorer(device)
    lm_model = BertForLM.from_pretrained(args.lm_model_dir)
    lm_model.to(device)
    lm_model.eval()
    for param in lm_model.parameters():
        param.requires_grad = False

    data = load_ext_sum_data()
    for ex in data:
        ex = preprocess(ex)
        src, segs, clss, src_sent_labels, src_txt, tgt_txt = ex
        if int(len(src_sent_labels) * args.insert_pos) == 0:
            # too short to insert collision into the article
            continue

        truth = [src_txt[j] for j in range(len(src_sent_labels)) if src_sent_labels[j] == 1]
        truth = ' '.join(truth)
        if args.nature:
            collision, score, rank = gen_natural_collision(ex, model, tokenizer, device, lm_model, eval_lm_model)
        else:
            collision, score, rank = gen_aggressive_collision(ex, model, tokenizer, device, lm_model)

        lm_perp = eval_lm_model.perplexity(collision)
        msg = f'Ground truth summary={truth}\n' \
              f'Collision={collision}\n' \
              f'Score={score}\n' \
              f'Rank={rank}\n' \
              f'LM perp={lm_perp.item()}\n'
        log(msg)


if __name__ == '__main__':
    args = parser.parse_args()
    main()
