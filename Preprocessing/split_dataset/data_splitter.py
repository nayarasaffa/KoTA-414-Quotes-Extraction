import json
import random
import argparse

class DataSplitter:
    def __init__(self, args):
        self.input_file = args.input_file
        self.train_ratio = args.train_ratio
        self.test_size = args.test_size
        self.data = self.load_data()
    
    def load_data(self):
        with open(self.input_file, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def split_data(self):
        random.shuffle(self.data)
        
        test_data = self.data[:self.test_size]
        remaining_data = self.data[self.test_size:]
        
        train_size = int(self.train_ratio * len(remaining_data))
        train_data = remaining_data[:train_size]
        validation_data = remaining_data[train_size:]
        
        return train_data, validation_data, test_data
    
    def save_data(self, train_file="train.json", validation_file="validation.json", test_file="test.json"):
        train_data, validation_data, test_data = self.split_data()
        
        with open(train_file, "w", encoding="utf-8") as f:
            json.dump(train_data, f, indent=2, ensure_ascii=False)
        
        with open(validation_file, "w", encoding="utf-8") as f:
            json.dump(validation_data, f, indent=2, ensure_ascii=False)
        
        with open(test_file, "w", encoding="utf-8") as f:
            json.dump(test_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Data successfully split: {len(train_data)} train, {len(validation_data)} validation, {len(test_data)} test")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split JSON dataset into train, validation, and test sets.")
    parser.add_argument(
        "-i", "--input_file", type=str, help="Path to the input file"
    )
    parser.add_argument(
        "--train_ratio", type=float, default=0.7, help="Proportion of data to use for training (default: 0.7)"
    )
    parser.add_argument(
        "--test_size", type=int, default=10, help="Number of articles to use for testing (default: 10)"
    )
    args = parser.parse_args()
    splitter = DataSplitter(args)
    splitter.save_data()
