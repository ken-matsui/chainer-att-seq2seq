import datetime
import numpy as np
from chainer import Chain, Variable, cuda, optimizer, optimizers, serializers
import chainer.functions as F
import chainer.links as L
import MeCab

from att_seq2seq.model import AttSeq2Seq
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
# ]

# 定数
EMBED_SIZE = 100
HIDDEN_SIZE = 100
BATCH_SIZE = 6 # ミニバッチ学習のバッチサイズ数
BATCH_COL_SIZE = 15
EPOCH_NUM = 50 # エポック数
N = len(data) # 教師データの数

# 教師データの読み込み
data_converter = DataConverter(batch_col_size=BATCH_COL_SIZE) # データコンバーター
data_converter.load(data) # 教師データ読み込み
vocab_size = len(data_converter.vocab) # 単語数

# モデルの宣言
model = AttSeq2Seq(vocab_size=vocab_size, embed_size=EMBED_SIZE, hidden_size=HIDDEN_SIZE, batch_col_size=BATCH_COL_SIZE)
# ネットワークファイルの読み込み
#network = "./att_seq2seq_network/*******************network"
#serializers.load_npz(network, model)
opt = optimizers.Adam()
opt.setup(model)
opt.add_hook(optimizer.GradientClipping(5))
if FLAG_GPU:
	model.to_gpu(0)
model.reset()

# 学習開始
print("Train")
st = datetime.datetime.now()
for epoch in range(EPOCH_NUM):
	# ミニバッチ学習
	perm = np.random.permutation(N) # ランダムな整数列リストを取得
	total_loss = 0
	for i in range(0, N, BATCH_SIZE):
		enc_words = data_converter.train_queries[perm[i:i+BATCH_SIZE]]
		dec_words = data_converter.train_responses[perm[i:i+BATCH_SIZE]]
		model.reset()
		loss = model(enc_words=enc_words, dec_words=dec_words, train=True)
		loss.backward()
		loss.unchain_backward()
		total_loss += loss.data
		opt.update()
	output_path = "./train/{}_{}.network".format(epoch+1, total_loss)
	serializers.save_npz(output_path, model)
	if (epoch+1)%10 == 0:
		ed = datetime.datetime.now()
		print("epoch:\t{}\ttotal loss:\t{}\ttime:\t{}".format(epoch+1, total_loss, ed-st))
		st = datetime.datetime.now()

print("\nPredict")
def predict(model, query):
	enc_query = data_converter.sentence2ids(query, train=False)
	dec_response = model(enc_words=enc_query, train=False)
	response = data_converter.ids2words(dec_response)
	print(query, "=>", response)

predict(model, "初めまして。")
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