import numpy as np
import torch
import os

from transformers import WEIGHTS_NAME
from parlai.agents.transformer.modules import TransformerEncoder, \
    TransformerMemNetModel, TransformerResponseWrapper, _normalize, get_n_positions_from_options
from parlai.agents.transformer.polyencoder import PolyEncoderModule
from parlai.core.dict import DictionaryAgent
from models.polyencoder.layers import TransformerAREncoder, LMPredictionHead
from models.polyencoder.config_polyencoder import load_poly_encoder_opt, \
    load_bi_encoder_opt, load_pretrained_poly_encoder_opt, load_pretrained_bi_encoder_opt
from collections import OrderedDict


class PolyEncoderTransformerEncoder(TransformerEncoder):
    def forward(self, input, positions=None, segments=None, inputs_embeds=None):
        if input is None:
            device = inputs_embeds.device
            input_shape = inputs_embeds.size()[:2]
            mask = torch.ones(input_shape, dtype=torch.bool).to(device)
        else:
            input_shape = input.shape
            device = input.device
            mask = input != self.padding_idx

        if positions is None:
            positions = (mask.cumsum(dim=1, dtype=torch.int64) - 1).clamp_(min=0)

        tensor = self.embeddings(input) if inputs_embeds is None else inputs_embeds
        if self.embeddings_scale:
            tensor = tensor * np.sqrt(self.dim)

        if positions.max().item() > self.n_positions:
            raise ValueError(
                'You are inputting a sequence of {x} length, but only have '
                '--n-positions {y}. Set --truncate or increase --n-positions'.format(
                    x=positions.max().item(), y=self.n_positions
                )
            )
        position_embs = self.position_embeddings(positions).expand_as(tensor)
        tensor = tensor + position_embs

        if self.n_segments >= 1:
            if segments is None:
                segments = torch.zeros(input_shape, dtype=torch.long, device=device)
            tensor = tensor + self.segment_embeddings(segments)

        if self.variant == 'xlm':
            tensor = _normalize(tensor, self.norm_embeddings)

        # --dropout on the embeddings
        tensor = self.dropout(tensor)

        tensor *= mask.unsqueeze(-1).type_as(tensor)

        if getattr(self.layers, 'is_model_parallel', False):
            # factored out for readability. It is equivalent to the other condition
            tensor = self._apply_model_parallel(tensor, mask)
        else:
            for i in range(self.n_layers):
                tensor = self.layers[i](tensor, mask)

        if self.variant == 'prelayernorm':
            tensor = _normalize(tensor, self.norm_embeddings)
        tensor *= self.output_scaling
        if self.reduction_type == 'first':
            return tensor[:, 0, :]
        elif self.reduction_type == 'max':
            return tensor.max(dim=1)[0]
        elif self.reduction_type == 'mean':
            divisor = mask.float().sum(dim=1).unsqueeze(-1).clamp(min=1).type_as(
                tensor)
            output = tensor.sum(dim=1) / divisor
            return output
        elif self.reduction_type is None or 'none' in self.reduction_type:
            return tensor, mask
        else:
            raise ValueError("Can't handle --reduction-type {}".format(self.reduction_type))


def _build_encoder(
        opt,
        dictionary,
        embedding=None,
        padding_idx=None,
        reduction_type='mean',
        n_positions=1024,
        n_segments=0,
):
    n_layers = (
        opt['n_encoder_layers']
        if opt.get('n_encoder_layers', -1) > 0
        else opt['n_layers']
    )
    return PolyEncoderTransformerEncoder(
        n_heads=opt['n_heads'],
        n_layers=n_layers,
        embedding_size=opt['embedding_size'],
        ffn_size=opt['ffn_size'],
        vocabulary_size=len(dictionary),
        embedding=embedding,
        dropout=opt['dropout'],
        attention_dropout=opt['attention_dropout'],
        relu_dropout=opt['relu_dropout'],
        padding_idx=padding_idx,
        learn_positional_embeddings=opt['learn_positional_embeddings'],
        embeddings_scale=opt['embeddings_scale'],
        reduction_type=reduction_type,
        n_positions=n_positions,
        n_segments=n_segments,
        activation=opt['activation'],
        variant=opt['variant'],
        output_scaling=opt['output_scaling'],
    )


class PolyEncoderModel(PolyEncoderModule):
    def __init__(self, opt, dict_, null_idx):
        super(PolyEncoderModel, self).__init__(opt, dict_, null_idx)
        self.encoder_ctxt = self.get_encoder(
            opt=opt,
            dict_=dict_,
            null_idx=null_idx,
            reduction_type=None,
            for_context=True,
        )
        self.encoder_cand = self.get_encoder(
            opt=opt,
            dict_=dict_,
            null_idx=null_idx,
            reduction_type=opt['reduction_type'],
            for_context=False,
        )

    def get_encoder(self, opt, dict_, null_idx, reduction_type,
                    for_context: bool):
        n_positions = get_n_positions_from_options(opt)
        embeddings = self._get_embeddings(dict_=dict_, null_idx=null_idx, embedding_size=opt['embedding_size'])
        return PolyEncoderTransformerEncoder(
            n_heads=opt['n_heads'],
            n_layers=opt['n_layers'],
            embedding_size=opt['embedding_size'],
            ffn_size=opt['ffn_size'],
            vocabulary_size=len(dict_),
            embedding=embeddings,
            dropout=opt['dropout'],
            attention_dropout=opt['attention_dropout'],
            relu_dropout=opt['relu_dropout'],
            padding_idx=null_idx,
            learn_positional_embeddings=opt['learn_positional_embeddings'],
            embeddings_scale=opt['embeddings_scale'],
            reduction_type=reduction_type,
            n_positions=n_positions,
            n_segments=opt.get('n_segments', 2),
            activation=opt['activation'],
            variant=opt['variant'],
            output_scaling=opt['output_scaling'],
        )


class PretrainedPolyEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        opt = load_poly_encoder_opt()
        d = DictionaryAgent(opt)
        self.opt = opt
        self.model = PolyEncoderModel(opt, d, d[d.null_token])

    def get_input_embeddings(self):
        return self.model.encoder_cand.embeddings

    def forward(self, ctxt_input_ids=None, ctxt_inputs_embeds=None,
                cand_input_ids=None, cand_inputs_embeds=None,
                return_scores=True):
        outputs = ()
        if ctxt_input_ids is not None or ctxt_inputs_embeds is not None:
            ctxt_hiddens, ctxt_masks = self.model.encoder_ctxt(ctxt_input_ids, inputs_embeds=ctxt_inputs_embeds)
            outputs = outputs + (ctxt_hiddens, ctxt_masks,)

        if cand_input_ids is not None or cand_inputs_embeds is not None:
            cand_hiddens = self.model.encoder_cand(cand_input_ids, inputs_embeds=cand_inputs_embeds)
            outputs = outputs + (cand_hiddens,)

        if return_scores and len(outputs) == 3:
            scores = self.score(*outputs)
            outputs = (scores,) + outputs

        return outputs

    def score(self, ctxt_hiddens, ctxt_masks, cand_hiddens):
        bsz = ctxt_hiddens.size(0)
        dim = ctxt_hiddens.size(2)

        if self.model.type == 'codes':
            ctxt_rep = self.model.attend(
                self.model.code_attention,
                queries=self.model.codes.repeat(bsz, 1, 1),
                keys=ctxt_hiddens,
                values=ctxt_hiddens,
                mask=ctxt_masks,
            )
            ctxt_rep_mask = ctxt_rep.new_ones(bsz, self.model.n_codes).byte()
        elif self.model.type == 'n_first':
            # Expand the output if it is not long enough
            if ctxt_hiddens.size(1) < self.model.n_codes:
                difference = self.model.n_codes - ctxt_hiddens.size(1)
                extra_rep = ctxt_hiddens.new_zeros(bsz, difference, dim)
                ctxt_rep = torch.cat([ctxt_hiddens, extra_rep], dim=1)
                extra_mask = ctxt_masks.new_zeros(bsz, difference)
                ctxt_rep_mask = torch.cat([ctxt_masks, extra_mask], dim=1)
            else:
                ctxt_rep = ctxt_hiddens[:, 0: self.model.n_codes, :]
                ctxt_rep_mask = ctxt_masks[:, 0: self.model.n_codes]
        else:
            raise ValueError(self.model.type)

        if bsz > 1:
            cand_hiddens = torch.cat([cand_hiddens.unsqueeze(0)] * bsz, 0)
        else:
            cand_hiddens = cand_hiddens.unsqueeze(0)

        ctxt_final_rep = self.model.attend(self.model.attention, cand_hiddens, ctxt_rep, ctxt_rep, ctxt_rep_mask)
        scores = torch.sum(ctxt_final_rep * cand_hiddens, 2)
        return scores

    @classmethod
    def from_pretrained(cls):
        opt = load_poly_encoder_opt()
        model_file = opt['model_file']
        state_dict = torch.load(model_file, map_location='cpu')['model']
        self = cls()
        self.model.load_state_dict(state_dict)
        return self


class BiEncoderModel(TransformerMemNetModel):
    def __init__(self, opt, dictionary):
        super().__init__(opt, dictionary)
        n_positions = get_n_positions_from_options(opt)
        self.context_encoder = _build_encoder(
            opt,
            dictionary,
            self.embeddings,
            self.pad_idx,
            reduction_type=self.reduction_type,
            n_positions=n_positions,
            n_segments=self.n_segments,
        )

        if opt.get('share_encoders'):
            self.cand_encoder = TransformerResponseWrapper(self.context_encoder, self.context_encoder.out_dim)
        else:
            if not self.share_word_embedding:
                cand_embeddings = self.cand_embeddings
            else:
                cand_embeddings = self.embeddings
            self.cand_encoder = _build_encoder(
                opt,
                dictionary,
                cand_embeddings,
                self.pad_idx,
                n_positions=n_positions,
                reduction_type=self.reduction_type,
                n_segments=self.n_segments,
            )

        # build memory encoder
        if opt.get('wrap_memory_encoder', False):
            self.memory_transformer = TransformerResponseWrapper(self.context_encoder, self.context_encoder.out_dim)
        else:
            self.memory_transformer = self.context_encoder


class PretrainedBiEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        opt = load_bi_encoder_opt()
        d = DictionaryAgent(opt)
        self.opt = opt
        self.model = BiEncoderModel(opt, d)

    def get_input_embeddings(self):
        return self.model.cand_embeddings

    def forward(self, ctxt_input_ids=None, ctxt_inputs_embeds=None,
                cand_input_ids=None, cand_inputs_embeds=None,
                return_scores=True):
        outputs = ()
        if ctxt_input_ids is not None or ctxt_inputs_embeds is not None:
            ctxt_hiddens = self.model.context_encoder(ctxt_input_ids, inputs_embeds=ctxt_inputs_embeds)
            outputs = outputs + (ctxt_hiddens,)

        if cand_input_ids is not None or cand_inputs_embeds is not None:
            cand_hiddens = self.model.cand_encoder(cand_input_ids, inputs_embeds=cand_inputs_embeds)
            outputs = outputs + (cand_hiddens,)
        if return_scores and len(outputs) == 2:
            scores = self.score(*outputs)
            outputs = (scores,) + outputs
        return outputs

    def score(self, context_h, cands_h):
        # possibly normalize the context and candidate representations
        if self.opt['normalize_sent_emb']:
            context_h = context_h / context_h.norm(2, dim=1, keepdim=True)
            cands_h = cands_h / cands_h.norm(2, dim=1, keepdim=True)
        return torch.matmul(context_h, cands_h.t())

    @classmethod
    def from_pretrained(cls):
        opt = load_bi_encoder_opt()
        model_file = opt['model_file']
        state_dict = torch.load(model_file, map_location='cpu')['model']
        self = cls()
        self.model.load_state_dict(state_dict)
        return self


class PolyEncoderLM(torch.nn.Module):
    def __init__(self, opt, encoder_name='encoder_cand'):
        super().__init__()
        n_positions = get_n_positions_from_options(opt)
        d = DictionaryAgent(opt)
        e = torch.nn.Embedding(len(d), opt['embedding_size'], d[d.null_token])
        torch.nn.init.normal_(e.weight, mean=0, std=opt['embedding_size'] ** -0.5)
        torch.nn.init.constant_(e.weight[d[d.null_token]], 0)

        self.opt = opt
        self.vocab_size = len(d)
        encoder_cand = TransformerAREncoder(
            n_heads=opt['n_heads'],
            n_layers=opt['n_layers'],
            embedding_size=opt['embedding_size'],
            ffn_size=opt['ffn_size'],
            embedding=e,
            dropout=opt['dropout'],
            attention_dropout=opt['attention_dropout'],
            relu_dropout=opt['relu_dropout'],
            padding_idx=d[d.null_token],
            learn_positional_embeddings=opt['learn_positional_embeddings'],
            embeddings_scale=opt['embeddings_scale'],
            n_positions=n_positions,
            n_segments=opt.get('n_segments', 2),
            activation=opt['activation'],
            variant=opt['variant'],
            output_scaling=opt['output_scaling'],
        )
        self.encoder_name = encoder_name
        setattr(self, encoder_name, encoder_cand)
        self.cls = LMPredictionHead(opt, len(d))

    @property
    def enc(self):
        return getattr(self, self.encoder_name)

    def get_input_embeddings(self):
        return self.enc.embeddings

    def save_pretrained(self, save_directory):
        assert os.path.isdir(save_directory), "Saving path should be a directory" \
                                              " where the model and configuration can be saved"
        # Only save the model itself if we are using distributed training
        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(save_directory, WEIGHTS_NAME)
        torch.save(self.state_dict(), output_model_file)

    @classmethod
    def from_pretrained(cls, model_type='poly', checkpoint=None):
        if model_type == 'poly':
            opt = load_pretrained_poly_encoder_opt()
            encoder_name = 'encoder'
        else:
            raise ValueError(model_type)

        if checkpoint is None:
            model_file = opt['model_file']
            state_dict = torch.load(model_file, map_location='cpu')['model']
        else:
            model_file = os.path.join(checkpoint, WEIGHTS_NAME)
            state_dict = torch.load(model_file, map_location='cpu')

        self = cls(opt, encoder_name)
        common_state_dict = OrderedDict()
        for key in self.state_dict():
            if key in state_dict:
                common_state_dict[key] = state_dict[key]
            if not key.startswith('cls') and key not in state_dict:
                raise ValueError(f'Weight not found in pretrained model for {key}')
        self.load_state_dict(common_state_dict, strict=False)
        if hasattr(self, 'cls'):
            self.cls.decoder.weight = self.get_input_embeddings().weight
        return self

    @staticmethod
    def prepare_inputs_for_generation(input_ids, past):
        # only last token for inputs_ids if past is defined in kwargs
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)
        return {"input_ids": input_ids, "past": past}

    @staticmethod
    def enforce_repetition_penalty_(lprobs, batch_size, num_beams, prev_output_tokens, repetition_penalty):
        for i in range(batch_size * num_beams):
            for previous_token in set(prev_output_tokens[i].tolist()):
                # if score < 0 then repetition penalty has to
                # multiplied to reduce the previous token probability
                if lprobs[i, previous_token] < 0:
                    lprobs[i, previous_token] *= repetition_penalty
                else:
                    lprobs[i, previous_token] /= repetition_penalty

    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = []
        for layer_past in past:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` and `mems` is at 2nd position
            # print(layer_past.shape)
            reordered_layer_past = [layer_past[:, i].unsqueeze(1).clone().detach() for i in beam_idx]
            reordered_layer_past = torch.cat(reordered_layer_past, dim=1)
            # check that shape matches
            assert reordered_layer_past.shape == layer_past.shape
            reordered_past.append(reordered_layer_past)
        past = tuple(reordered_past)
        return past

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            labels=None,
            one_hot_labels=None,
            past=None,
            inputs_embeds=None,
    ):
        outputs = self.enc(
            input_ids,
            attention_mask=attention_mask,
            segments=token_type_ids,
            position_ids=position_ids,
            past=past,
            inputs_embeds=inputs_embeds
        )
        sequence_output = outputs[0]
        if hasattr(self, 'cls'):
            prediction_scores = self.cls(sequence_output)
        else:
            prediction_scores = torch.nn.functional.linear(sequence_output, self.get_input_embeddings().weight)
        # Add hidden states and attention if they are here
        outputs = (prediction_scores,) + outputs[1:]

        # we are doing next-token prediction;
        # shift prediction scores and input ids by one
        if one_hot_labels is not None:
            prediction_scores = prediction_scores[:, :-1, :].contiguous()
            lm_labels = one_hot_labels[:, 1:, :].contiguous()
            nll = -torch.log_softmax(prediction_scores, -1)
            ltr_lm_loss = torch.sum(nll * lm_labels, -1).mean()
            outputs = (ltr_lm_loss,) + outputs
        elif labels is not None:
            prediction_scores = prediction_scores[:, :-1, :].contiguous()
            lm_labels = labels[:, 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss()
            ltr_lm_loss = loss_fct(prediction_scores.view(-1, self.vocab_size), lm_labels.view(-1))
            outputs = (ltr_lm_loss,) + outputs
        return outputs
