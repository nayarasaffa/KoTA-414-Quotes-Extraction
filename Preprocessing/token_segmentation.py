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
        return re.findall(r'\w+-?\w+|[^\w\s]', sentence, re.UNICODE)

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

            # Setelah kutip pembuka → tidak ada spasi
            elif i > 0 and tokens[i - 1] in {'"', "'"} and is_quotes:
                text += token
        
            else:
                text += " " + token  # Token biasa diberi spasi sebelum mereka

        return text.strip()

    def segment_sentences(self, tokens):
        text = self.adjust_spacing(tokens)

        # print(text)
        
        sentence_segmenter = SentenceSegmentation()
        sentences = sentence_segmenter.get_sentences(tokens)

        # for i, sent in enumerate(sentences):
        #     print(f"{i+1}. {sent}")
        
        return sentences

    def map_tokens_to_sentences(self, tokens, tags, sentences):
        segmented_tokens = [token for sentence in sentences for token in sentence]
        
        for i, (t_orig, t_seg) in enumerate(zip(tokens, segmented_tokens)):
            if t_orig != t_seg:
                print(f"❌ Token tidak cocok pada indeks {i}: original = '{t_orig}', segmented = '{t_seg}'")
                break

        output_lines = []
        idx = 0  # index untuk tokens dan tags

        for sentence in sentences:
            current_len = len(sentence)
            for _ in range(current_len):
                token = tokens[idx]
                tag = tags[idx]
                output_lines.append(f"{token}\t{tag}")
                idx += 1
            output_lines.append("")  # baris kosong untuk pemisah antar kalimat

        return output_lines

    def write_tsv(self, file_path, lines):
        with open(file_path, 'w', encoding='utf-8') as f:
            for line in lines:
                f.write(line + '\n')

    def process(self):
        print(f"Processing {self.input_file} ...")
        tokens, tags = self.read_tsv(self.input_file)
        sentences = self.segment_sentences(tokens)

        # print(f"Total tokens     : {len(tokens)}")
        # print(f"Total tags       : {len(tags)}")
        # print(f"Total sentences  : {len(sentences)}")
        # print(f"Total tokens in sentences: {sum(len(s) for s in sentences)}")
        
        segmented_data = self.map_tokens_to_sentences(tokens, tags, sentences)
        self.write_tsv(self.output_file, segmented_data)
        print(f"Segmented data saved to {self.output_file}")