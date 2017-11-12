import numpy as np
from chainer import cuda

from att_seq2seq.model import AttSeq2Seq
from att_seq2seq.trainer import Trainer
from data_utils.converter import DataConverter

# GPUのセット
FLAG_GPU = False # GPUを使用するかどうか
if FLAG_GPU: # numpyかcuda.cupyか
	xp = cuda.cupy
	cuda.get_device(0).use()
else:
	xp = np

# 学習
# 教師データ
DATA_PATH = './data/'
with open(DATA_PATH+'input.txt', 'r') as fin, open(DATA_PATH+'output.txt', 'r') as fout:
	inp, out = fin.readlines(), fout.readlines()
data = list(map(lambda l: list(l), list(zip(inp, out))))
# data = [
# 	["初めまして。", "初めまして。よろしくお願いします。"],
# 	["どこから来たんですか？", "日本から来ました。"],
# 	["日本のどこに住んでるんですか？", "東京に住んでいます。"],
# 	["仕事は何してますか？", "私は会社員です。"],
# 	["お会いできて嬉しかったです。", "私もです！"],
# 	["おはよう。", "おはようございます。"],
# 	["いつも何時に起きますか？", "6時に起きます。"],
# 	["朝食は何を食べますか？", "たいていトーストと卵を食べます。"],
# 	["朝食は毎日食べますか？", "たまに朝食を抜くことがあります。"],
# 	["野菜をたくさん取っていますか？", "毎日野菜を取るようにしています。"],
# 	["週末は何をしていますか？", "友達と会っていることが多いです。"],
# 	["どこに行くのが好き？", "私たちは渋谷に行くのが好きです。"]
# ] # sample data.

# 定数
EMBED_SIZE = 100
HIDDEN_SIZE = 100
BATCH_SIZE = 6 # ミニバッチ学習のバッチサイズ数
BATCH_COL_SIZE = 15
EPOCH_NUM = 50 # エポック数
teacher_num = len(data) # 教師データの数

# 教師データの読み込み
data_converter = DataConverter(batch_col_size=BATCH_COL_SIZE) # データコンバーター
data_converter.load(data) # 教師データ読み込み
vocab_size = len(data_converter.vocab) # 単語数

# モデルの宣言
model = AttSeq2Seq(vocab_size=vocab_size,
				   embed_size=EMBED_SIZE,
				   hidden_size=HIDDEN_SIZE,
				   batch_col_size=BATCH_COL_SIZE)
# 学習開始
print("Train")
trainer = Trainer(model)
trainer.fit(queries=data_converter.train_queries,
			responses=data_converter.train_responses,
			teacher_num=teacher_num,
			epoch_num=EPOCH_NUM,
			batch_size=BATCH_SIZE)


print("\nPredict")
def predict(model, query):
	enc_query = data_converter.sentence2ids(query, train=False)
	dec_response = model(enc_words=enc_query, train=False)
	response = data_converter.ids2words(dec_response)
	print(query, "=>", response)

predict(model, "こんにちは")
predict(model, "どこから来たんですか？")
predict(model, "日本のどこに住んでるんですか？")
predict(model, "仕事は何してますか？")
predict(model, "お会いできて嬉しかったです。")
predict(model, "おはよう。")
predict(model, "いつも何時に起きますか？")
predict(model, "朝食は何を食べますか？")
predict(model, "朝食は毎日食べますか？")
predict(model, "野菜をたくさん取っていますか？")
predict(model, "週末は何をしていますか？")
predict(model, "どこに行くのが好き？")
predict(model, "今日はどこへ行くのですか？")