# coding: utf8
import sys
import os
import collections
import random
import pickle

class DataReader(object):
    def __init__(self, vocab_path, data_path, vocab_size=10000, batch_size=64, max_seq_len=48):
        """ init
        """
        self._batch_size = batch_size
        self._vocab_size = vocab_size
        self._max_seq_len = max_seq_len
        if not os.path.exists(vocab_path):
            self._word_to_id = self._build_vocab(data_path)
            with open(vocab_path, "w") as ofs:
                pickle.dump(self._word_to_id, ofs)
        else:
            with open(vocab_path, "r") as ifs:
                self._word_to_id = pickle.load(ifs)
        self._data = self._build_data(data_path)

    def _build_vocab(self, filename):
        with open(filename, "r") as ifs:
            data = ifs.read().replace("\n", " ").split()
        counter = collections.Counter(data)
        count_pairs = counter.most_common(self._vocab_size - 2)

        words, _ = list(zip(*count_pairs))
        word_to_id = dict(zip(words, range(2, len(words) + 2)))
        word_to_id["<pad>"] = 0
        word_to_id["<unk>"] = 1
        print("vocab words num: ", len(word_to_id))
        return word_to_id

    def _build_data(self, filename, is_shuffle=True):
        with open(filename, "r") as ifs:
            lines = ifs.readlines()
            data = list(map(lambda x: x.strip().split("\t"), lines))
            random.shuffle(data)
        return data

    def _padding_batch(self, batch):
        for idx, line in enumerate(batch[0]):
            if len(line) > self._max_seq_len:
                batch[0][idx] = line[:self._max_seq_len]
            else:
                batch[0][idx] = line + [self._word_to_id["<pad>"]] * (self._max_seq_len - len(line)) 
        return batch

    def batch_generator(self):
        curr_size = 0
        batch = [[], []]
        for line in self._data:
            curr_size += 1
            text, label = line
            text_ids = [self._word_to_id.get(x, self._word_to_id["<unk>"]) for x in text.split()]
            batch[0].append(text_ids)
            if label == "0":
                batch[1].append([1, 0])
            else:
                batch[1].append([0, 1])
            if curr_size >= self._batch_size:
                yield self._padding_batch(batch)
                batch = [[], []]
                curr_size = 0
        if curr_size > 0:
            yield self._padding_batch(batch)

if __name__ == "__main__":
    reader = DataReader("data/vocab.pkl", "data/train.txt")
    for batch in reader.batch_generator():
        print(batch)
