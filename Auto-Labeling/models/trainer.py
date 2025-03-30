# New code add early stopping
import torch
from spacy.lang.id import Indonesian

class Trainer(object):
    def __init__(self, model, data):
        self.model = model
        self.data = data
    
    def infer(self, sentence, true_tags=None):
        self.model.eval()
        nlp = Indonesian()
        tokens = [token.text for token in nlp(sentence)]
        max_word_len = max([len(token) for token in tokens])
        numericalized_tokens = [self.data.word_field.vocab.stoi[token.lower()] for token in tokens]
        numericalized_chars = []
        char_pad_id = self.data.char_pad_idx
        for token in tokens:
            numericalized_chars.append(
                [self.data.char_field.vocab.stoi[char] for char in token]
                + [char_pad_id for _ in range(max_word_len - len(token))]
            )
        unk_idx = self.data.word_field.vocab.stoi[self.data.word_field.unk_token]
        unks = [t for t, n in zip(tokens, numericalized_tokens) if n == unk_idx]
        token_tensor = torch.as_tensor(numericalized_tokens).unsqueeze(-1)
        char_tensor = torch.as_tensor(numericalized_chars).unsqueeze(0)
        predictions, _, attn_weight = self.model(token_tensor, char_tensor)
        predicted_tags = [self.data.tag_field.vocab.itos[t] for t in predictions[0]]
        max_len_token = max([len(token) for token in tokens] + [len('word')])
        max_len_tag = max([len(tag) for tag in predicted_tags] + [len('pred')])
        # print(f"{'word'.ljust(max_len_token)}\t{'unk'.ljust(max_len_token)}\t{'pred tag'.ljust(max_len_tag)}"
        #       + ("\ttrue tag" if true_tags else ""))
        # for i, token in enumerate(tokens):
        #     is_unk = "✓" if token in unks else ""
        #     print(f"{token.ljust(max_len_token)}\t{is_unk.ljust(max_len_token)}\t{predicted_tags[i].ljust(max_len_tag)}"
        #           + (f"\t{true_tags[i]}" if true_tags else ""))
        return tokens, predicted_tags, unks
    
    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        print(f"✅ Model loaded from {model_path}")