from parlai.core.dict import DictionaryAgent
from transformers import PreTrainedTokenizer
from models.polyencoder.config_polyencoder import load_poly_encoder_opt


class PolyEncoderTokenizer(PreTrainedTokenizer):
    def __init__(self, **kwargs):
        opt = load_poly_encoder_opt()
        self.dict = DictionaryAgent(opt)
        super().__init__(
            unk_token=self.dict.unk_token,
            pad_token=self.dict.null_token,
            cls_token=self.dict.start_token,
            sep_token=self.dict.end_token, **kwargs,
        )

    def get_vocab(self):
        return self.dict.tok2ind

    def save_vocabulary(self, save_directory):
        pass

    @property
    def vocab_size(self):
        return len(self.dict.tok2ind)

    def _tokenize(self, text, **kwargs):
        return self.dict.tokenize(str(text))

    def _convert_token_to_id(self, token):
        return self.dict[token]

    def _convert_id_to_token(self, index):
        return self.dict.ind2tok.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens):
        out_string = self.dict.bpe.decode(tokens, token_ids=[], delimiter=' ')
        return out_string

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep + token_ids_1 + sep

    @classmethod
    def from_pretrained(cls, *inputs, **kwargs):
        return cls()
