import argparse
import numpy as np
from preprocessing import Preprocessing
from token_segmentation import TokenSegmentation

class Main:
    def __init__(self, args):
        self.input_file = f"dataset/{args.input_file}"
        self.output_file = f"dataset/{args.output_file}"
        self.check_dataset_flag = args.check_dataset

        super().__init__()

    def preprocessing(self, input_file, output_file):
        preprocessing = Preprocessing(input_file, output_file)
        preprocessing.process()

    def token_segmentation(self, input_file, output_file):
        token_segmentation = TokenSegmentation(input_file, output_file)
        token_segmentation.process()

    def check_dataset(self, input_file):
        with open(input_file, "r", encoding="utf-8") as f:
            sentences = f.read().strip().split("\n\n")
            token_counts = [len(sentence.split("\n")) for sentence in sentences]
        
        longest_sentence = max(sentences, key=lambda s: len(s.split("\n")))

        print(f"  Total Token: {sum(token_counts)}")
        print(f"  Rata-rata Token per Kalimat: {np.mean(token_counts)}")
        print(f"  Token Maksimum dalam Satu Kalimat: {np.max(token_counts)}\n")
        print(f"Kalimat dengan token terbanyak ({np.max(token_counts)} token):\n")
        print(longest_sentence)

    def main(self):
        self.preprocessing(self.input_file, self.output_file)
        self.token_segmentation(self.input_file if self.token_segmentation_flag else self.output_file, self.output_file)
        if self.check_dataset_flag:
            self.check_dataset(self.input_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocessing text")
    parser.add_argument(
        "-i", "--input_file", type=str, help="Path to the input file"
    )
    parser.add_argument(
        "-o", "--output_file", type=str, help="Path to the output file"
    )
    parser.add_argument(
        "-c", "--check_dataset", action="store_true", help="Check dataset."
    )
    args = parser.parse_args()
    main = Main(args)
    main.main()