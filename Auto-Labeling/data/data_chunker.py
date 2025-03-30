import json
import argparse
import os

class DataChunker:
    def __init__(self, args):
        self.input_file = args.input_file
        self.chunk_size = args.chunk_size
        self.data = self.load_data()
    
    def load_data(self):
        with open(self.input_file, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def split_data(self):
        return [self.data[i:i + self.chunk_size] for i in range(0, len(self.data), self.chunk_size)]
    
    def save_chunks(self):
        os.makedirs(self.output_dir, exist_ok=True)
        chunks = self.split_data()

        file_name, file_ext = os.path.splitext(self.input_file)
        output_file = f"{file_name}_{idx+1}{file_ext}"

        
        for idx, chunk in enumerate(chunks):
            with open(f"{output_file}_{idx+1}.json", "w", encoding="utf-8") as f:
                json.dump(chunk, f, indent=4, ensure_ascii=False)
            print(f"File {output_file}_{idx+1}.json telah dibuat.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split a JSON dataset into smaller chunks.")
    parser.add_argument(
        "-i", "--input_file", type=str, help="Path to the input file"
    )
    parser.add_argument(
        "--chunk_size", type=int, default=10, help="Number of articles per chunk (default: 10)"
    )
    args = parser.parse_args()
    chunker = DataChunker(args)
    chunker.save_chunks()