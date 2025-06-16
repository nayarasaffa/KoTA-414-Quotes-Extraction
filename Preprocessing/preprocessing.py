import argparse
import numpy as np
from data_cleansing import DataCleansing
from token_segmentation import TokenSegmentation
from split_dataset.data_splitter import DataSplitter

class Preprocessing:
    def __init__(self, args):
        self.input_file = f"dataset/{args.input_file}" if args.input_file is not None else None
        self.output_file = f"dataset/{args.output_file}" if args.output_file is not None else None
        self.check_dataset_flag = args.check_dataset

        self.split_dataset = args.split_dataset
        self.train_ratio = args.train_ratio
        self.test_size = args.test_size

        super().__init__()

    def preprocessing(self, input_file, output_file):
        preprocessing = DataCleansing(input_file, output_file)
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
        print(f"  Token Maksimum dalam Satu Kalimat: {np.max(token_counts)}")
        print(f"  Token Minimum dalam Satu Kalimat: {np.min(token_counts)}")
        print(f"Kalimat dengan token terbanyak ({np.max(token_counts)} token):\n")
        print(longest_sentence)

    def main(self):
        if self.split_dataset:
            splitter = DataSplitter(self.input_file, self.train_ratio, self.test_size)
            splitter.save_data()
        if (self.input_file != None) and (self.output_file != None):
            self.preprocessing(self.input_file, self.output_file)
            self.token_segmentation(self.output_file, self.output_file)
            if self.check_dataset_flag:
                self.check_dataset(self.output_file)
        elif self.check_dataset_flag:
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
    parser.add_argument(
        "--train_ratio", type=float, default=0.7, help="Proportion of data to use for training (default: 0.7)"
    )
    parser.add_argument(
        "--test_size", type=int, default=10, help="Number of articles to use for testing (default: 10)"
    )
    parser.add_argument(
        "--split_dataset", action="store_true", help="Split dataset."
    )
    args = parser.parse_args()
    main = Preprocessing(args)
    main.main()