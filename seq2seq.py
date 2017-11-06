# coding:utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import math
import os
import random
import sys
import time

import numpy as np
import tensorflow as tf
import tensorflow.python.platform
from tensorflow.contrib import rnn
from tensorflow.contrib import legacy_seq2seq
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from tqdm import tqdm

import parse
import mecab
import split

# tf.app.flags.DEFINE_string('変数名', 'デフォルト値', """説明文""")
# python script.py --'変数名' ok => tf.app.flags.FLAGS.'変数名' == 'ok'
tf.app.flags.DEFINE_boolean("parse", False, "Run a parse if this is set to True")
tf.app.flags.DEFINE_boolean("decode", False, "Set to True for interactive decoding.")
FLAGS = tf.app.flags.FLAGS


def seq2seq():
	if FLAGS.decode:
		print("decode()")
		# decode()
	else:
		print("train()")
		# train()

def main(_):
	if FLAGS.parse:
		print("parse...")
		parse.main()
		print("mecab...")
		mecab.main()
		print("split...")
		split.main()
		print("seq2seq...")
		seq2seq()
	else:
		print("skip parse")
		print("skip mecab")
		print("skip split")
		print("seq2seq...")
		seq2seq()

if __name__ == '__main__':
	tf.app.run()