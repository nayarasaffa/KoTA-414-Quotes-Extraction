import json
import argparse

from models.corpus import Corpus
from models.bilstm import BiLSTM
from models.trainer import Trainer
from concurrent.futures import ThreadPoolExecutor, as_completed
from idsentsegmenter.sentence_segmentation import SentenceSegmentation

class AutoLabeling:
    def __init__(self, args):
        self.trainer = self.load_model()
        self.sentence_segmenter = SentenceSegmentation()

        self.input_file = args.input_file
        self.output_file = args.output_file

        super().__init__()

    def load_model(self):
        # Load Corpus Model
        print("Loading Corpus Model...")
        corpus = Corpus (
            input_folder = "dataset",
            min_word_freq = 3,
            batch_size = 64,
            wv_file = "models/pretrain/embeddings/wiki.id.case.model"
        )

        # Load BiLSTM Model
        print("Loading BiLSTM Model...")
        bilstm = BiLSTM(
            input_dim=len(corpus.word_field.vocab),
            embedding_dim=400,
            char_emb_dim=25,
            char_input_dim=len(corpus.char_field.vocab),
            char_cnn_filter_num=5,
            char_cnn_kernel_size=3,
            hidden_dim=64,
            output_dim=len(corpus.tag_field.vocab),
            lstm_layers=2,
            attn_heads=16,
            emb_dropout=0.5,
            cnn_dropout=0.25,
            lstm_dropout=0.1,
            attn_dropout=0.25,
            fc_dropout=0.25,
            word_pad_idx=corpus.word_pad_idx,
            char_pad_idx=corpus.char_pad_idx,
            tag_pad_idx=corpus.tag_pad_idx
        )

        # Load Trainer Model
        print("Loading Trainer Model...")
        trainer = Trainer(
            model=bilstm,
            data=corpus
        )
        trainer.load_model("models/model/bilstm_model.pt")

        return trainer
    
    def clean_tag(self, tag:str) -> str:
        return tag.split('-')[-1] if '-' in tag else tag
    
    def read_json(self, file_path):
        print(f"Reading file {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def write_json(self, file_path, data):
        print(f"Writing file {file_path}")
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    def process_data(self, data, index):
        try:
            print(f"Processing data {index+1}")
            text = data['text']
            sentences = self.sentence_segmenter.get_sentences(text)
            result = []

            start_offset = 0
            for sentence in sentences:
                if len(sentence) > 0:
                    tokens, predicted_tags, _ = self.trainer.infer(sentence)
                    for token, tag in zip(tokens, predicted_tags):
                        if tag != 'O':
                            entity_start = text.find(token, start_offset)
                            entity_end = entity_start + len(token)
                            start_offset = entity_end

                            result.append({
                                "value": {
                                    'start': entity_start,
                                    'end': entity_end,
                                    'text': token,
                                    'labels': [self.clean_tag(tag)],
                                },
                                "from_name": "label",
                                "to_name": "text",
                                "type": "labels",
                                "origin": "manual"
                            })

            return {
                "annotations": [{"result": result}],
                'data': {"text": text}
            }
        
        except Exception as e:
            print(f"❌ Error pada data {index+1}: {str(e)}")
            return None
    
    def main(self):
        # Read text from json
        datas = self.read_json(self.input_file)

        predictions = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_index = {executor.submit(self.process_data, data, i): i for i, data in enumerate(datas)}

            for future in as_completed(future_to_index):
                result = future.result()
                if result is not None:
                    predictions.append(result)

        self.write_json(self.output_file, predictions)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocessing text")
    parser.add_argument(
        "-i", "--input_file", type=str, help="Path to the input file"
    )
    parser.add_argument(
        "-o", "--output_file", type=str, help="Path to the output file"
    )
    args = parser.parse_args()
    auto_labeling = AutoLabeling(args)
    auto_labeling.main()