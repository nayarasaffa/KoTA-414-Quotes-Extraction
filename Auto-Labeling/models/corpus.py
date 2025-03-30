import torch
import gensim
from collections import Counter
from torchtext.vocab import Vocab
from torchtext.datasets import SequenceTaggingDataset
from torchtext.data import Field, NestedField, BucketIterator

class Corpus(object):

    def __init__(self, input_folder, min_word_freq, batch_size, wv_file=None):
        # list all the fields
        self.word_field = Field(lower=True)  # [sent len, batch_size]
        self.tag_field = Field(unk_token=None)  # [sent len, batch_size]

        ### BEGIN MODIFIED SECTION: CHARACTER EMBEDDING ###
        self.char_nesting_field = Field(tokenize=list)
        self.char_field = NestedField(self.char_nesting_field)  # [batch_size, sent len, word len]

        # create dataset using built-in parser from torchtext
        self.train_dataset, self.val_dataset, self.test_dataset = SequenceTaggingDataset.splits(
            path=input_folder,
            train="train-70.tsv",
            validation="val-30.tsv",
            test="test-data.tsv",
            fields=(
                (("word", "char"), (self.word_field, self.char_field)),
                ("tag", self.tag_field)
            )
        )
        ### END MODIFIED SECTION ###

        # convert fields to vocabulary list
        if wv_file:
            # Load pretrained FastText model
            self.wv_model = gensim.models.FastText.load(wv_file)
            self.embedding_dim = self.wv_model.vector_size

            # Build vocab based on FastText model
            word_freq = {word: self.wv_model.wv.get_vecattr(word, "count") for word in self.wv_model.wv.index_to_key}
            word_counter = Counter(word_freq)
            self.word_field.vocab = Vocab(word_counter, min_freq=min_word_freq)

            # Mapping each vector/embedding from FastText model to word_field vocabs
            vectors = []
            for word, idx in self.word_field.vocab.stoi.items():
                if word in self.wv_model.wv:
                    vectors.append(torch.as_tensor(self.wv_model.wv[word].tolist()))
                else:
                    vectors.append(torch.zeros(self.embedding_dim))

            self.word_field.vocab.set_vectors(
                stoi=self.word_field.vocab.stoi,
                vectors=vectors,  # List of vector embeddings, ordered according to word_field.vocab
                dim=self.embedding_dim
            )
        else:
            self.word_field.build_vocab(self.train_dataset.word, min_freq=min_word_freq)

        # build vocab for tag and characters
        self.char_field.build_vocab(self.train_dataset.char)  # NEWLY ADDED
        self.tag_field.build_vocab(self.train_dataset.tag)

        # create iterator for batch input
        self.train_iter, self.val_iter, self.test_iter = BucketIterator.splits(
            datasets=(self.train_dataset, self.val_dataset, self.test_dataset),
            batch_size=batch_size
        )

        # prepare padding index to be ignored during model training/evaluation
        self.word_pad_idx = self.word_field.vocab.stoi[self.word_field.pad_token]
        self.char_pad_idx = self.char_field.vocab.stoi[self.char_field.pad_token]  # NEWLY ADDED
        self.tag_pad_idx = self.tag_field.vocab.stoi[self.tag_field.pad_token]