# coding: utf-8

import chainer.functions as F
import chainer.links as L
from chainer import Chain, Variable, cuda
import numpy as xp # xp これxpだとまずい
# # GPUのセット # TODO: どこで行う？
# FLAG_GPU = False # GPUを使用するかどうか
# if FLAG_GPU: # numpyかcuda.cupyか
# 	xp = cuda.cupy
# 	cuda.get_device(0).use()
# else:
# 	xp = np

__all__ = ['AttSeq2Seq']

# LSTMエンコーダークラス
class LSTMEncoder(Chain):
	def __init__(self, vocab_size, embed_size, hidden_size):
		'''
		Encoderのインスタンス化
		:param vocab_size: 使われる単語の種類数
		:param embed_size: 単語をベクトル表現した際のサイズ
		:param hidden_size: 隠れ層のサイズ
		'''
		super(LSTMEncoder, self).__init__(
			xe = L.EmbedID(vocab_size, embed_size, ignore_label=-1),
			eh = L.Linear(embed_size, 4 * hidden_size),
			hh = L.Linear(hidden_size, 4 * hidden_size)
		)

	def __call__(self, x, c, h):
		'''
		Encoderの計算
		:param x: one-hotな単語
		:param c: 内部メモリ
		:param h: 隠れ層
		:return: 次の内部メモリ、次の隠れ層
		'''
		e = F.tanh(self.xe(x))
		return F.lstm(c, self.eh(e) + self.hh(h))

# Attention Model + LSTMデコーダークラス
class AttLSTMDecoder(Chain):
	def __init__(self, vocab_size, embed_size, hidden_size):
		'''
		Attention ModelのためのDecoderのインスタンス化
		:param vocab_size: 語彙数
		:param embed_size: 単語ベクトルのサイズ
		:param hidden_size: 隠れ層のサイズ
		'''
		super(AttLSTMDecoder, self).__init__(
			ye = L.EmbedID(vocab_size, embed_size, ignore_label=-1), # 単語を単語ベクトルに変換する層
			eh = L.Linear(embed_size, 4 * hidden_size), # 単語ベクトルを隠れ層の4倍のサイズのベクトルに変換する層
			hh = L.Linear(hidden_size, 4 * hidden_size), # Decoderの中間ベクトルを隠れ層の4倍のサイズのベクトルに変換する層
			fh = L.Linear(hidden_size, 4 * hidden_size), # 順向きEncoderの中間ベクトルの加重平均を隠れ層の4倍のサイズのベクトルに変換する層
			bh = L.Linear(hidden_size, 4 * hidden_size), # 順向きEncoderの中間ベクトルの加重平均を隠れ層の4倍のサイズのベクトルに変換する層
			he = L.Linear(hidden_size, embed_size), # 隠れ層サイズのベクトルを単語ベクトルのサイズに変換する層
			ey = L.Linear(embed_size, vocab_size) # 単語ベクトルを語彙数サイズのベクトルに変換する層
		)

	def __call__(self, y, c, h, f, b):
		'''
		Decoderの計算
		:param y: Decoderに入力する単語
		:param c: 内部メモリ
		:param h: Decoderの中間ベクトル
		:param f: Attention Modelで計算された順向きEncoderの加重平均
		:param b: Attention Modelで計算された逆向きEncoderの加重平均
		:return: 語彙数サイズのベクトル、更新された内部メモリ、更新された中間ベクトル
		'''
		e = F.tanh(self.ye(y)) # 単語を単語ベクトルに変換
		c, h = F.lstm(c, self.eh(e) + self.hh(h) + self.fh(f) + self.bh(b)) # 単語ベクトル、Decoderの中間ベクトル、順向きEncoderのAttention、逆向きEncoderのAttentionを使ってLSTM
		t = self.ey(F.tanh(self.he(h))) # LSTMから出力された中間ベクトルを語彙数サイズのベクトルに変換する
		return t, c, h

# Attentionモデルクラス
class Attention(Chain):
	def __init__(self, hidden_size):
		'''
		Attentionのインスタンス化
		:param hidden_size: 隠れ層のサイズ
		'''
		super(Attention, self).__init__(
			fh = L.Linear(hidden_size, hidden_size), # 順向きのEncoderの中間ベクトルを隠れ層サイズのベクトルに変換する線形結合層
			bh = L.Linear(hidden_size, hidden_size), # 逆向きのEncoderの中間ベクトルを隠れ層サイズのベクトルに変換する線形結合層
			hh = L.Linear(hidden_size, hidden_size), # Decoderの中間ベクトルを隠れ層サイズのベクトルに変換する線形結合層
			hw = L.Linear(hidden_size, 1), # 隠れ層サイズのベクトルをスカラーに変換するための線形結合層
		)
		self.hidden_size = hidden_size # 隠れ層のサイズを記憶

	def __call__(self, fs, bs, h):
		'''
		Attentionの計算
		:param fs: 順向きのEncoderの中間ベクトルが記録されたリスト
		:param bs: 逆向きのEncoderの中間ベクトルが記録されたリスト
		:param h: Decoderで出力された中間ベクトル
		:return: 順向きのEncoderの中間ベクトルの加重平均と逆向きのEncoderの中間ベクトルの加重平均
		'''
		batch_size = h.data.shape[0] # ミニバッチのサイズを記憶
		ws = [] # ウェイトを記録するためのリストの初期化
		sum_w = Variable(xp.zeros((batch_size, 1), dtype='float32')) # ウェイトの合計値を計算するための値を初期化
		# Encoderの中間ベクトルとDecoderの中間ベクトルを使ってウェイトの計算
		for f, b in zip(fs, bs):
			w = F.tanh(self.fh(f)+self.bh(b)+self.hh(h)) # 順向きEncoderの中間ベクトル、逆向きEncoderの中間ベクトル、Decoderの中間ベクトルを使ってウェイトの計算
			w = F.exp(self.hw(w)) # softmax関数を使って正規化する
			ws.append(w) # 計算したウェイトを記録
			sum_w += w
		# 出力する加重平均ベクトルの初期化
		att_f = Variable(xp.zeros((batch_size, self.hidden_size), dtype='float32'))
		att_b = Variable(xp.zeros((batch_size, self.hidden_size), dtype='float32'))
		for f, b, w in zip(fs, bs, ws):
			w /= sum_w # ウェイトの和が1になるように正規化
			# ウェイト * Encoderの中間ベクトルを出力するベクトルに足していく
			att_f += F.reshape(F.batch_matmul(f, w), (batch_size, self.hidden_size))
			att_b += F.reshape(F.batch_matmul(b, w), (batch_size, self.hidden_size))
		return att_f, att_b

# Attention Sequence to Sequence Modelクラス
class AttSeq2Seq(Chain):
	def __init__(self, vocab_size, embed_size, hidden_size, batch_col_size):
		'''
		Attention + Seq2Seqのインスタンス化
		:param vocab_size: 語彙数のサイズ
		:param embed_size: 単語ベクトルのサイズ
		:param hidden_size: 隠れ層のサイズ
		'''
		super(AttSeq2Seq, self).__init__(
			f_encoder = LSTMEncoder(vocab_size, embed_size, hidden_size), # 順向きのEncoder
			b_encoder = LSTMEncoder(vocab_size, embed_size, hidden_size), # 逆向きのEncoder
			attention = Attention(hidden_size), # Attention Model
			decoder = AttLSTMDecoder(vocab_size, embed_size, hidden_size) # Decoder
		)
		self.vocab_size = vocab_size
		self.embed_size = embed_size
		self.hidden_size = hidden_size
		self.decode_max_size = batch_col_size # デコードはEOSが出力されれば終了する、出力されない場合の最大出力語彙数
		# 順向きのEncoderの中間ベクトル、逆向きのEncoderの中間ベクトルを保存するためのリストを初期化
		self.fs = []
		self.bs = []

	def __call__(self, enc_words, dec_words=None, train=True):
		'''
		順伝播の計算を行う関数
		:param enc_words: 発話文の単語を記録したリスト
		:param dec_words: 応答文の単語を記録したリスト
		:param train: 学習か予測か
		:return: 計算した損失の合計 or 予測したデコード文字列
		'''
		enc_words = enc_words.T
		if train:
			dec_words = dec_words.T
		batch_size = len(enc_words[0]) # バッチサイズを記録
		self.reset() # model内に保存されている勾配をリセット
		enc_words = [Variable(xp.array(row, dtype='int32')) for row in enc_words] # 発話リスト内の単語をVariable型に変更
		self.encode(enc_words, batch_size) # エンコードの計算
		t = Variable(xp.array([0 for _ in range(batch_size)], dtype='int32')) # <eos>をデコーダーに読み込ませる
		loss = Variable(xp.zeros((), dtype='float32')) # 損失の初期化
		ys = [] # デコーダーが生成する単語を記録するリスト
		# デコーダーの計算
		if train: # 学習の場合は損失を計算する
			for w in dec_words:
				y = self.decode(t) # 1単語ずつをデコードする
				t = Variable(xp.array(w, dtype='int32')) # 正解単語をVariable型に変換
				loss += F.softmax_cross_entropy(y, t) # 正解単語と予測単語を照らし合わせて損失を計算
			return loss
		else: # 予測の場合はデコード文字列を生成する
			for i in range(self.decode_max_size):
				y = self.decode(t)
				y = xp.argmax(y.data) # 確率で出力されたままなので、確率が高い予測単語を取得する
				ys.append(y)
				t = Variable(xp.array([y], dtype='int32'))
				if y == 0: # EOSを出力したならばデコードを終了する
					break
			return ys

	def encode(self, words, batch_size):
		'''
		Encoderの計算
		:param words: 入力で使用する単語記録されたリスト
		:param batch_size: ミニバッチのサイズ
		:return:
		'''
		# 内部メモリ、中間ベクトルの初期化
		c = Variable(xp.zeros((batch_size, self.hidden_size), dtype='float32'))
		h = Variable(xp.zeros((batch_size, self.hidden_size), dtype='float32'))
		# 順向きのEncoderの計算
		for w in words:
			c, h = self.f_encoder(w, c, h)
			self.fs.append(h) # 計算された中間ベクトルを記録
		# 内部メモリ、中間ベクトルの初期化
		c = Variable(xp.zeros((batch_size, self.hidden_size), dtype='float32'))
		h = Variable(xp.zeros((batch_size, self.hidden_size), dtype='float32'))
		# 逆向きのEncoderの計算
		for w in reversed(words):
			c, h = self.b_encoder(w, c, h)
			self.bs.insert(0, h) # 計算された中間ベクトルを記録
		# 内部メモリ、中間ベクトルの初期化
		self.c = Variable(xp.zeros((batch_size, self.hidden_size), dtype='float32'))
		self.h = Variable(xp.zeros((batch_size, self.hidden_size), dtype='float32'))

	def decode(self, w):
		'''
		Decoderの計算
		:param w: Decoderで入力する単語
		:return: 予測単語
		'''
		att_f, att_b = self.attention(self.fs, self.bs, self.h)
		t, self.c, self.h = self.decoder(w, self.c, self.h, att_f, att_b)
		return t

	def reset(self):
		'''
		インスタンス変数を初期化する
		Encoderの中間ベクトルを記録するリストの初期化
		'''
		self.fs = []
		self.bs = []
		# 勾配の初期化
		self.zerograds()
