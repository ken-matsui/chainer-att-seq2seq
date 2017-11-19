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

	def sentence2words(self, sentence):
		'''
		文章を単語に分解する
		:param sentence: 文章文字列
		:return: 単語ごとに分割した配列
		'''
		# Natural Language API
		# The text to analyze
		# document = language.types.Document(
		# 	content=sentence,
		# 	type=language.enums.Document.Type.PLAIN_TEXT
		# )
		# # Detects syntax in the document. You can also analyze HTML with:
		# #   document.type == enums.Document.Type.HTML
		# tokens = self.client.analyze_syntax(document).tokens

		# sentence_words = []
		# for token in tokens:
		# 	w = token.text.content # 単語
		# 	if len(w) != 0: # 不正文字は省略
		# 		sentence_words.append(w)
		# sentence_words.append("<eos>") # 最後にvocabに登録している<eos>を代入する
		# return sentence_words

	def sentence2ids(self, sentence):
		'''
		文章を単語IDのNumpy配列に変換して返却する
		:param sentence: 文章文字列
		:return: 単語IDのNumpy配列
		'''
		ids = [] # 単語IDに変換して格納する配列
		sentence_words = self.sentence2words(sentence) # 文章を単語に分解する
		for word in sentence_words:
			if word in self.vocab: # 単語辞書に存在する単語ならば、IDに変換する
				ids.append(self.vocab.index(word))
			else: # 単語辞書に存在しない単語ならば、<unk>のIDに変換する
				ids.append(self.vocab.index("<unk>"))
		ids = xp.array([ids], dtype="int32")
		return ids

	def ids2words(self, ids):
		'''
		予測時に、単語IDのNumpy配列を単語に変換して返却する
		:param ids: 単語IDのNumpy配列
		:return: 単語の配列
		'''
		words = [] # 単語を格納する配列
		for i in ids: # 順番に単語IDを単語辞書から参照して単語に変換する
			words.append(vocab[i])
		return words
