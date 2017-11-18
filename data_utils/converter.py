# coding: utf-8

import os

from chainer import cuda
import numpy as np
import MeCab

FLAG_GPU = False
if FLAG_GPU:
	xp = cuda.cupy
	cuda.get_device(0).use()
else:
	xp = np

# データ変換クラスの定義
class DataConverter:
	def __init__(self, batch_col_size):
		'''
		クラスの初期化
		:param batch_col_size: 学習時のミニバッチ単語数サイズ
		'''
		self.mecab = MeCab.Tagger('-d /usr/local/lib/mecab/dic/mecab-ipadic-neologd')
		self.vocab = {}
		self.batch_col_size = batch_col_size

	def load(self, path):
		'''
		学習時に、教師データを読み込んでミニバッチサイズに対応したNumpy配列に変換する
		:param path: dataがあるディレクトリ
		'''
		# 対話データを取り出す
		with open(path + 'query.txt', 'r') as fq, open(path + 'response.txt', 'r') as fr:
			que, res = fq.readlines(), fr.readlines()
		data = list(zip(que, res))
		self.teacher_num = len(data)

		# 単語辞書データを取り出す
		with open(path + 'vocab.txt', 'r') as f:
			lines = f.readlines()
		for i, line in enumerate(lines):
			self.vocab[line.replace('\n', '')] = i

		# 教師データのID化と整理
		queries, responses = [], []
		for d in data:
			query, response = d[0], d[1] #  エンコード文、デコード文
			queries.append(self.sentence2ids(sentence=query, train=True, sentence_type="query"))
			responses.append(self.sentence2ids(sentence=response, train=True, sentence_type="response"))
		self.train_queries = xp.vstack(queries)
		self.train_responses = xp.vstack(responses)

	def sentence2words(self, sentence):
		'''
		文章を単語に分解する
		:param sentence: 文章文字列
		:return: 単語ごとに分割した配列
		'''
		sentence_words = []
		for m in self.mecab.parse(sentence).split("\n"): # 形態素解析で単語に分解する
			w = m.split("\t")[0].lower() # 単語
			if (len(w) == 0) or (w is "eos"): # 不正文字、EOSは省略
				continue
			sentence_words.append(w)
		sentence_words.append("<eos>") # 最後にvocabに登録している<eos>を代入する
		return sentence_words

	def sentence2ids(self, sentence, train=True, sentence_type="query"):
		'''
		文章を単語IDのNumpy配列に変換して返却する
		:param sentence: 文章文字列
		:param train: 学習用かどうか
		:sentence_type: 学習用でミニバッチ対応のためのサイズ補填方向をクエリー・レスポンスで変更するため"query"or"response"を指定　
		:return: 単語IDのNumpy配列
		'''
		ids = [] # 単語IDに変換して格納する配列
		sentence_words = self.sentence2words(sentence) # 文章を単語に分解する
		for word in sentence_words:
			if word in self.vocab: # 単語辞書に存在する単語ならば、IDに変換する
				ids.append(self.vocab[word])
			else: # 単語辞書に存在しない単語ならば、<unk>のIDに変換する
				ids.append(self.vocab["<unk>"])
		# 学習時は、ミニバッチ対応のため、単語数サイズを調整してNumpy変換する
		if train:
			if sentence_type == "query": # クエリーの場合は前方にミニバッチ単語数サイズになるまで-1を補填する
				while len(ids) > self.batch_col_size: # ミニバッチ単語サイズよりも大きければ、ミニバッチ単語サイズになるまで先頭から削る
					ids.pop(0)
				ids = xp.array([-1]*(self.batch_col_size-len(ids))+ids, dtype="int32")
			elif sentence_type == "response": # レスポンスの場合は後方にミニバッチ単語数サイズになるまで-1を補填する
				while len(ids) > self.batch_col_size: # ミニバッチ単語サイズよりも大きければ、ミニバッチ単語サイズになるまで末尾から削る
					ids.pop()
				ids = xp.array(ids+[-1]*(self.batch_col_size-len(ids)), dtype="int32")
		else: # 予測時は、そのままNumpy変換する
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
			words.append(list(self.vocab.keys())[list(self.vocab.values()).index(i)])
		return words
