# coding: utf-8

from chainer import serializers, cuda
from google.cloud import language
import MeCab

class Decoder(object):
	def __init__(self, model, npz, vocab, decode_max_size, flag_gpu=False):
		self.model = model
		self.vocab = vocab
		self.decode_max_size = decode_max_size

		# self.client = language.LanguageServiceClient()
		self.mecab = MeCab.Tagger('-d /usr/local/lib/mecab/dic/mecab-ipadic-neologd')

		serializers.load_npz(npz, self.model)
		if flag_gpu:
			import cupy as cp
			self.xp = cp
			self.model.to_gpu(0)
			cuda.get_device(0).use()
		else:
			import numpy as np
			self.xp = np

	def __call__(self, query):
		# モデルの勾配などをリセット
		self.model.reset()
		# userからの入力文をIDに変換
		enc_query = self.sentence2ids(query)
		# Numpy配列に変換
		enc_query = self.xp.array([enc_query], dtype="int32")
		enc_query = enc_query.T
		# エンコード時のバッチサイズ
		encode_batch_size = len(enc_query[0])
		# エンコードの計算
		self.model.encode(enc_query, encode_batch_size)
		# <eos>をデコーダーに読み込ませる
		t = self.xp.array([0] * encode_batch_size, dtype='int32')
		# デコーダーが生成する単語IDリスト
		ys = []
		for i in range(self.decode_max_size):
			y = self.model.decode(t)
			y = self.xp.argmax(y.data) # 確率で出力されたままなので、確率が高い予測単語を取得する
			ys.append(y)
			t = self.xp.array([y], dtype='int32')
			if y == 0: # <EOS>を出力したならばデコードを終了する
				break
		# IDから，文字列に変換する
		response = self.ids2words(ys)

		if "<eos>" in response: # 最後の<eos>を回避
			res = "".join(response[0:-1])
		else: # 含んでない時もある．(出力wordサイズが，15を超えた時？？？)
			res = "".join(response)
		return res

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

		sentence_words = []
		# for token in tokens:
		# 	w = token.text.content # 単語
		# 	if len(w) != 0: # 不正文字は省略
		# 		sentence_words.append(w)
		# sentence_words.append("<eos>") # 最後にvocabに登録している<eos>を代入する
		for m in self.mecab.parse(sentence).split("\n"):
			w = m.split("\t")[0].lower()
			if (len(w) == 0) or (w == "eos"):
				continue
			sentence_words.append(w)
		sentence_words.append("<eos>")

		return sentence_words

	def sentence2ids(self, sentence):
		'''
		文章を単語IDの配列に変換して返却する
		:param sentence: 文章文字列
		:return: 単語IDの配列
		'''
		ids = [] # 単語IDに変換して格納する配列
		sentence_words = self.sentence2words(sentence) # 文章を単語に分解する
		for word in sentence_words:
			if word in self.vocab: # 単語辞書に存在する単語ならば、IDに変換する
				ids.append(self.vocab.index(word))
			else: # 単語辞書に存在しない単語ならば、<unk>のIDに変換する
				ids.append(self.vocab.index("<unk>"))
		return ids

	def ids2words(self, ids):
		'''
		予測時に、単語IDのNumpy配列を単語に変換して返却する
		:param ids: 単語IDのNumpy配列
		:return: 単語の配列
		'''
		words = [] # 単語を格納する配列
		for i in ids: # 順番に単語IDを単語辞書から参照して単語に変換する
			words.append(self.vocab[int(i)])
		return words