# coding: utf-8

import os

from chainer import cuda
# from google.cloud import language

FLAG_GPU = False
if FLAG_GPU:
	import cupy as cp
	xp = cp
	cuda.get_device(0).use()
else:
	import numpy as np
	xp = np

# データ変換クラスの定義
class DataConverter:
	def __init__(self, batch_col_size):
		'''
		クラスの初期化
		:param batch_col_size: 学習時のミニバッチ単語数サイズ
		'''
		# Instantiates a client
		# self.client = language.LanguageServiceClient()
		# 単語辞書は，配列の添字をIDとして使う
		self.vocab = []
		self.batch_col_size = batch_col_size

	def load(self, path):
		'''
		学習時に、教師データを読み込んでミニバッチサイズに対応したNumpy配列に変換する
		:param path: dataがあるディレクトリ
		'''
		# 対話データ(ID版)を取り出す
		with open(path + 'query_id.txt', 'r') as fqid, open(path + 'response_id.txt', 'r') as frid:
			queid, resid = fqid.readlines(), frid.readlines()
		self.teacher_num = len(list(zip(queid, resid)))
		queries, responses = [], []
		for q, r in zip(queid, resid):
			# ミニバッチ対応のため，単語数サイズを調整してNumpy変換する
			query = list(map(int, q.replace('\n', '').split(',')))
			queries.append(self.batch_ids(query))
			respo = list(map(int, q.replace('\n', '').split(',')))
			responses.append(self.batch_ids(respo))
		self.train_queries = xp.vstack(xp.array(queries))
		self.train_responses = xp.vstack(xp.array(responses))

		# 単語辞書データを取り出す
		with open(path + 'vocab.txt', 'r') as f:
			lines = f.readlines()
		self.vocab = list(map(lambda s: s.replace("\n", ""), lines))

	def batch_ids(self, ids):
		if len(ids) > self.batch_col_size: # ミニバッチ単語サイズになるように先頭から削る
			del ids[0:len(ids) - self.batch_col_size]
		else: # ミニバッチ単語サイズになるように先頭に付け足す
			ids = ([-1] * (self.batch_col_size - len(ids))) + ids
		return ids
