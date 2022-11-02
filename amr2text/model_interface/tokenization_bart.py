# coding:utf-8
# this is a simplified version of "https://github.com/SapienzaNLP/spring/blob/main/spring_amr/tokenization_bart.py"

import regex as re
from transformers import BartTokenizer
from common.constant import raw_special_tokens, recategorizations


class AMRBartTokenizer(BartTokenizer):
    INIT = 'Ġ'
    
    def __init__(self, vocab_file, merges_file, errors="replace", bos_token="<s>", eos_token="</s>", sep_token="</s>", cls_token="<s>", unk_token="<unk>", pad_token="<pad>", mask_token="<mask>", add_prefix_space=False, **kwargs):
        super().__init__(vocab_file, merges_file, errors, bos_token, eos_token, sep_token, cls_token, unk_token, pad_token, mask_token, add_prefix_space, **kwargs)
        self.modified = 0
        self.recategorizations = set(recategorizations)
        self.patterns = re.compile(r""" ?<[a-z]+:?\d*>| ?:[^\s]+|'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        
    @classmethod
    def from_pretrained(cls, pretrained_model_path, *args, **kwargs):
        inst = super().from_pretrained(pretrained_model_path, *args, **kwargs)
        inst.init_amr_vocabulary()
        return inst
    
    def init_amr_vocabulary(self):
        self.old_enc_size = old_enc_size = len(self.encoder)
        tokens = [t for t in raw_special_tokens if t not in self.encoder]

        for i, t in enumerate(tokens, start=old_enc_size):
            self.encoder[t] = i

        self.encoder = {k: i for i, (k,v) in enumerate(sorted(self.encoder.items(), key=lambda x: x[1]))}
        self.decoder = {v: k for k, v in sorted(self.encoder.items(), key=lambda x: x[1])}
        self.modified = len(tokens)

        self.amr_bos_token = "<AMR>"
        self.amr_bos_token_id = self.encoder[self.amr_bos_token]
        self.amr_eos_token = "</AMR>"
        self.amr_eos_token_id = self.encoder[self.amr_eos_token]
        print(f"Added {self.modified} AMR tokens")
    
    def _tokenize(self, text):
        """ Tokenize a string. Modified in order to handle sentences with recategorization pointers"""
        bpe_tokens = []
        for tok_span in text.lstrip().split(' '):
            tok_span = tok_span.strip()
            recats = tok_span.rsplit('_', 1)
            if len(recats) == 2 and recats[0] in self.recategorizations and ('_' + recats[1]) in self.encoder:
                bpe_tokens.extend([self.INIT + recats[0], '_' + recats[1]])
            else:
                for token in re.findall(self.pat, ' ' + tok_span):
                    token = "".join(
                        self.byte_encoder[b] for b in token.encode("utf-8")
                    )   # Maps all our bytes to unicode strings, avoiding controle tokens of the BPE (spaces in our case)
                    bpe_tokens.extend(bpe_token for bpe_token in self.bpe(token).split(" "))

        return bpe_tokens

    def _tok_bpe(self, token):
        tokk = []
        tok = token.strip()
        recats = tok.rsplit('_', 1)
        if len(recats) == 2 and recats[0] in self.recategorizations and ('_' + recats[1]) in self.encoder:
            tokk.extend([self.INIT + recats[0], '_' + recats[1]])
        else:
            for tok in self.patterns.findall(' ' + token):
                tok = "".join(
                    self.byte_encoder[b] for b in tok.encode("utf-8"))
                toks = self.bpe(tok).split(' ')
                tokk.extend(toks)
        return tokk

    def tokenize_amr(self, amr_tokens):
        bpe_tokens = []
        for i, tokk in enumerate(amr_tokens):
            is_in_enc = self.INIT + tokk in self.encoder
            is_rel = tokk.startswith(':') and len(tokk) > 1
            is_spc = tokk.startswith('<') and tokk.endswith('>')
            is_of = tokk.startswith(':') and tokk.endswith('-of')
            is_frame = re.match(r'.+-\d\d', tokk) is not None

            if tokk.startswith('"') and tokk.endswith('"'):                 # dealing with examples like "The_United_Kingdom_of_xxx"
                tokk = tokk[1:-1].replace('_', ' ')
                bpe_toks = [self.INIT + "<lit>"]
                bpe_toks += self._tok_bpe(tokk)
                bpe_toks.append(self.INIT + "</lit>")

            elif (is_rel or is_spc or is_frame or is_of):
                if is_in_enc:
                    bpe_toks = [self.INIT + tokk]
                elif is_frame:
                    bpe_toks = self._tok_bpe(tokk[:-3]) + [tokk[-3:]]
                elif is_of:
                    rel = tokk[:-3]
                    if self.INIT + rel in self.encoder:
                        bpe_toks = [self.INIT + rel, '-of']
                    else:
                        bpe_toks = [self.INIT + ':'] + self._tok_bpe(rel[1:]) + ['-of']
                elif is_rel:
                    bpe_toks = [self.INIT + ':'] + self._tok_bpe(tokk[1:])
                else:
                    print("tok:", tokk)
                    print(f"is_rel:{is_rel}, is_spc:{is_spc}, is_frame:{is_frame}, is_of:{is_of}")
                    exit()
                    raise
            else:
                if is_in_enc:
                    bpe_toks = [self.INIT + tokk]
                else:
                    bpe_toks = self._tok_bpe(tokk)

            bpe_tokens.append(bpe_toks)
        bpe_tokens = [b for bb in bpe_tokens for b in bb]
        bpe_token_ids = [self.encoder.get(b, self.unk_token_id) for b in bpe_tokens]
        return bpe_token_ids