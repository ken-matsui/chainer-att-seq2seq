# coding: utf-8

import os

from chainer import serializers

from att_seq2seq.model import AttSeq2Seq
from att_seq2seq.trainer import Trainer


def load_vocab():
    with open("./data/vocab.txt", 'r') as f:
        lines = f.readlines()
    return list(map(lambda s: s.replace("\n", ""), lines))

def main():
    vocab = load_vocab()
    model = AttSeq2Seq(vocab_size=len(vocab),
                        embed_size=100,
                        hidden_size=100,
                        flag_gpu=False)
    serializers.load_npz("./models/80.npz", model)
    serializers.save_hdf5("./models/80.h5", model)

if __name__ == '__main__':
	main()
