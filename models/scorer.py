from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch


class GPT2PerpModel(GPT2LMHeadModel):
    def forward(
            self,
            input_ids=None,
            past=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
    ):
        transformer_outputs = self.transformer(
            input_ids,
            past=past,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)

        outputs = (lm_logits,) + transformer_outputs[1:]
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_masks = attention_mask[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss = loss.view(shift_labels.shape[0], -1) * shift_masks
            loss = torch.sum(loss, -1) / torch.sum(shift_masks, -1)
            outputs = (loss,) + outputs
        return outputs  # (loss), lm_logits, presents, (all hidden_states), (attentions)


class SentenceScorer(object):
    def __init__(self, device):
        self.lm_model = GPT2PerpModel.from_pretrained('gpt2')
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.lm_model.eval()
        self.lm_model.to(device)
        self.device = device
        for param in self.lm_model.parameters():
            param.requires_grad = False

    def perplexity(self, inputs):
        if isinstance(inputs, str):
            inputs = [inputs]
        inputs = self.tokenizer.batch_encode_plus(inputs, pad_to_max_length=True)
        attention_mask = torch.tensor(inputs['attention_mask'], device=self.device)
        inputs = torch.tensor(inputs['input_ids'], device=self.device)
        loss = torch.scalar_tensor(20.0)
        if inputs.shape[1] > 1:
            loss = self.lm_model(inputs, attention_mask=attention_mask, labels=inputs)[0].squeeze()
        return torch.exp(loss)
