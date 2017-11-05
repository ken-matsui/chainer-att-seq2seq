# coding:utf-8

import math
import os
import random
import sys
import time

import numpy as np
import tensorflow as tf
import tensorflow.python.platform
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from tqdm import tqdm

import parse
import mecab
import split
import seq2seq_model

tf.app.flags.DEFINE_boolean("parse", False, "Run a parse if this is set True")
FLAGS = tf.app.flags.FLAGS

def seq2seq():
	print("hoge")

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

if __name__ == '__main__':
	tf.app.run()