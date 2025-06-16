import re

class DataCleansing:
    def __init__(self, input_file, output_file):
        self.input_file = input_file
        self.output_file = output_file

    def convert_conll_to_tsv(self):
        """Mengubah format CoNLL menjadi TSV dengan dua kolom: token dan tag."""
        with open(self.input_file, 'r', encoding='utf-8') as infile, \
            open(self.output_file, 'w', encoding='utf-8') as outfile:

            for line in infile:
                line = line.strip()
                if not line:
                    continue

                parts = line.split()

                if len(parts) == 3:
                    continue

                token, tag = parts[0], parts[-1]

                if token == "-DOCSTART-":
                    continue

                token = self.normalize_quotes(token)

                outfile.write(f"{token}\t{tag}\n")

        print(f"File {self.input_file} berhasil dikonversi ke format TSV.")

    def normalize_quotes(self, text):
        """Mengganti tanda kutip miring dengan tanda kutip lurus."""
        text = re.sub(r'[“”]', '"', text)
        text = re.sub(r'[‘’]', "'", text)
        return text        
    
    def convert_bio_to_bilou(self):
        with open(self.output_file, 'r', encoding='utf-8') as infile:
            bio_lines = infile.readlines()

        bilou_lines = []
        sentence = []

        for line in bio_lines:
            if line.strip() == "":
                # Process the sentence when a blank line is encountered
                if sentence:
                    words, tags = zip(*sentence)
                    bilou_tags = []
                    i = 0
                    while i < len(tags):
                        tag = tags[i]
                        # print(f'i: {i}, tag: {tag}')
                        if tag.startswith("B-"):
                            label = tag[2:]
                            j = i + 1
                            while j < len(tags) and (tags[j] == f"I-{label}" or tags[j] == f"B-{label}"):
                                j += 1
                            if j - i == 1:
                                bilou_tags.append(f"U-{label}")
                            else:
                                bilou_tags.append(f"B-{label}")
                                for k in range(i + 1, j - 1):
                                    bilou_tags.append(f"I-{label}")
                                bilou_tags.append(f"L-{label}")
                            i = j
                        else:
                            bilou_tags.append(tag)
                            i += 1
                    for (word, _), tag in zip(sentence, bilou_tags):
                        bilou_lines.append(f"{word}\t{tag}\n")
                    bilou_lines.append("\n")
                    sentence = []
            else:
                parts = line.strip().split()
                word, tag = parts[0], parts[-1]
                sentence.append((word, tag))

        # Handle last sentence if file doesn't end with newline
        if sentence:
            words, tags = zip(*sentence)
            bilou_tags = []
            i = 0
            while i < len(tags):
                tag = tags[i]
                if tag.startswith("B-"):
                    label = tag[2:]
                    j = i + 1
                    while j < len(tags) and (tags[j] == f"I-{label}" or tags[j] == f"B-{label}"):
                        j += 1
                    if j - i == 1:
                        bilou_tags.append(f"U-{label}")
                    else:
                        bilou_tags.append(f"B-{label}")
                        for k in range(i + 1, j - 1):
                            bilou_tags.append(f"I-{label}")
                        bilou_tags.append(f"L-{label}")
                    i = j
                else:
                    bilou_tags.append(tag)
                    i += 1
            for (word, _), tag in zip(sentence, bilou_tags):
                bilou_lines.append(f"{word}\t{tag}\n")

        with open(self.output_file, 'w', encoding='utf-8') as outfile:
            outfile.writelines(bilou_lines)

        print(f"File {self.output_file} berhasil dikonversi ke skema BILOU.")

    def split_characters_from_tokens(self):
        """Memisahkan karakter yang menempel pada token."""
        with open(self.output_file, 'r', encoding='utf-8') as infile:
            lines = infile.readlines()
        
        tokens, tags = [], []
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 2:
                tokens.append("\n")
                tags.append("\n")
                continue
            tokens.append(parts[0])
            tags.append(parts[1])
        
        new_tokens, new_tags = self.preprocess_tokens(tokens, tags)
        
        with open(self.output_file, 'w', encoding='utf-8') as outfile:
            for token, tag in zip(new_tokens, new_tags):
                outfile.write(f"{token}\t{tag}\n")
    
    def preprocess_tokens(self, tokens, tags):
        """
        Memproses token dengan aturan pemisahan simbol dan penentuan tag.
        """
        new_tokens, new_tags = [], []

        for i in range(len(tokens)):
            token = tokens[i]
            tag = tags[i]
            split_tokens = self.split_token(token)

            # Atur ulang tag sesuai aturan yang diberikan
            for j, sub_token in enumerate(split_tokens):
                if j == 0:  # Token pertama tetap dengan tag asli
                    new_tokens.append(sub_token)
                    new_tags.append(tag)
                else:
                    # Jika masih dalam entitas yang sama, gunakan tag asli
                    if i + 1 < len(tokens) and tags[i + 1] == tag:
                        new_tags.append(tag)
                    else:
                        new_tags.append("O")  # Jika beda entitas, beri tag 'O'
                    new_tokens.append(sub_token)

        return new_tokens, new_tags

    def split_token(self, token):
        """
        Memisahkan simbol dari token utama.
        Contoh: '"2024."' -> ['"', '2024', '.']
        """
        parts = re.findall(r'\w+-?\w+|[^\w\s]', token, re.UNICODE)
        return parts

    def process(self):
        self.convert_conll_to_tsv()
        self.convert_bio_to_bilou()
        # self.split_characters_from_tokens()