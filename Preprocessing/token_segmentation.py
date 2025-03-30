import re
import csv
from idsentsegmenter.sentence_segmentation import SentenceSegmentation

class TokenSegmentation:
    def __init__(self, input_file, output_file):
        self.input_file = input_file
        self.output_file = output_file

    def read_tsv(self, file_path):
        tokens, tags = [], []
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
            for row in reader:
                if len(row) == 2:
                    token, tag = row[0].strip(), row[1].strip()
                    if token and tag:
                        tokens.append(token)
                        tags.append(tag)
        return tokens, tags
    
    def split_tokens(self, sentence):
        """Pisahkan semua kata dan simbol"""
        return re.findall(r"\w+|[^\w\s]", sentence, re.UNICODE)

    def adjust_spacing(self, tokens):
        """Menggabungkan token dengan aturan spasi yang benar"""
        text = ""
        is_quotes = False  # Menandai apakah sedang dalam kutipan

        for i, token in enumerate(tokens):
            # Token pertama langsung ditambahkan tanpa spasi
            if i == 0:
                text += token
                continue

            # Jika token adalah kutip (") atau (')
            if token in {'"', "'"}:
                if is_quotes:
                    text += token  # Kutip penutup, tidak ada spasi sebelum
                else:
                    text += " " + token  # Kutip pembuka, ada spasi sebelum
                is_quotes = not is_quotes  # Toggle status kutipan

            # Jika token adalah tanda baca (. , : ; ! ? )
            elif re.match(r'^[.,:;!?)]$', token):
                text += token  # Tidak ada spasi sebelum
            
            # Jika token adalah kurung pembuka '('
            elif re.match(r'^[(]$', token):
                text += " " + token  # Ada spasi sebelum
            
            # Jika token adalah teks biasa
            elif i > 0 and re.match(r'^[(]$', tokens[i - 1]):
                text += token  # Jika token sebelumnya adalah `(`, tidak ada spasi sebelum token sekarang

            else:
                text += " " + token  # Token biasa diberi spasi sebelum mereka

        return text.strip()

    def segment_sentences(self, tokens):
        text = self.adjust_spacing(tokens)

        # print(text)
        
        sentence_segmenter = SentenceSegmentation()
        sentences = sentence_segmenter.get_sentences(text)

        # for i, sent in enumerate(sentences):
        #     print(f"{i+1}. {sent}")
        
        return sentences

    def map_tokens_to_sentences(self, tokens, tags, sentences):
        token_idx = 0
        result = []
        
        for sentence in sentences:
            # print(f"Sentence in map tokens: {sentence}")
            sentence = sentence
            sentence_tokens = self.split_tokens(sentence)  # Tokenisasi ulang berdasarkan spasi
            for token in sentence_tokens:
                # print(f"Token in map tokens: {token}")
                while token_idx < len(tokens) and tokens[token_idx] != token:
                    token_idx += 1  # Cari token yang sesuai dalam daftar asli
                if token_idx < len(tokens):
                    result.append(f"{tokens[token_idx]}\t{tags[token_idx]}")
                    token_idx += 1
            result.append("")  # Tambahkan baris kosong sebagai pemisah antar kalimat
        
        return result

    def write_tsv(self, file_path, output_lines):
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(output_lines))

    def process(self):
        print(f"Processing {self.input_file} ...")
        tokens, tags = self.read_tsv(self.input_file)
        sentences = self.segment_sentences(tokens)
        segmented_data = self.map_tokens_to_sentences(tokens, tags, sentences)
        self.write_tsv(self.output_file, segmented_data)
        print(f"Segmented data saved to {self.output_file}")