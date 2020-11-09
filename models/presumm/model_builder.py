import torch
import torch.nn as nn
from transformers import BertModel, BertConfig
from models.presumm.encoder import Classifier, ExtTransformerEncoder


def get_generator(vocab_size, dec_hidden_size, device):
    gen_func = nn.LogSoftmax(dim=-1)
    generator = nn.Sequential(
        nn.Linear(dec_hidden_size, vocab_size),
        gen_func
    )
    generator.to(device)

    return generator


class Bert(nn.Module):
    def __init__(self):
        super(Bert, self).__init__()
        self.model = BertModel.from_pretrained('bert-base-uncased')

    def forward(self, x, segs, mask, x_embeds=None):
        top_vec, _ = self.model(input_ids=x, token_type_ids=segs, attention_mask=mask, inputs_embeds=x_embeds)
        return top_vec


class ExtSummarizer(nn.Module):
    def __init__(self, config, checkpoint=None):
        super(ExtSummarizer, self).__init__()
        self.bert = Bert()

        self.ext_layer = ExtTransformerEncoder(self.bert.model.config.hidden_size,
                                               config.ext_ff_size, config.ext_heads,
                                               config.ext_dropout, config.ext_layers)
        if config.encoder == 'baseline':
            bert_config = BertConfig(self.bert.model.config.vocab_size,
                                     hidden_size=config.ext_hidden_size,
                                     num_hidden_layers=config.ext_layers,
                                     num_attention_heads=config.ext_heads,
                                     intermediate_size=config.ext_ff_size)
            self.bert.model = BertModel(bert_config)
            self.ext_layer = Classifier(self.bert.model.config.hidden_size)

        if checkpoint is not None:
            self.load_state_dict(checkpoint, strict=True)

        if config.bert_max_pos > 512:
            my_pos_embeddings = nn.Embedding(config.bert_max_pos, self.bert.model.config.hidden_size)
            my_pos_embeddings.weight.data[:512] = self.bert.model.embeddings.position_embeddings.weight.data
            my_pos_embeddings.weight.data[512:] = \
                self.bert.model.embeddings.position_embeddings.weight.data[-1][None, :].repeat(
                    config.bert_max_pos - 512, 1)
            self.bert.model.embeddings.position_embeddings = my_pos_embeddings

    def forward(self, src, segs, clss, mask_src, mask_cls, src_embeds=None,
                output_logits=False):
        top_vec = self.bert(src, segs, mask_src, src_embeds)
        sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss]
        sents_vec = sents_vec * mask_cls[:, :, None].float()
        sent_scores = self.ext_layer(sents_vec, mask_cls, output_logits).squeeze(-1)
        return sent_scores
