# coding:utf-8

import os
import re
from urllib.request import urlopen

import MeCab
from bs4 import BeautifulSoup

MECAB = MeCab.Tagger('-d /usr/local/lib/mecab/dic/mecab-ipadic-neologd')

WORDS_URL = "http://svn.sourceforge.jp/svnroot/slothlib/CSharp/Version1/SlothLib/NLP/Filter/StopWord/word/Japanese.txt"
with urlopen(WORDS_URL) as words_file:
	STOP_WORDS = words_file.read().decode('utf-8').split('\r\n')

RE_TIME = re.compile(r"[0-9]?[0-9]:[0-9]?[0-9]")
RE_TAB = re.compile(r"\t")
RE_IGNORE = re.compile(r"(\[Photo\]|\[Sticker\]|\[Video\]|\[Albums\]|\[File\]|☎)")
RE_URL = re.compile(r"(https?|ftp)(:\/\/[-_\.!~*\'()a-zA-Z0-9;\/?:\@&=\+\$,%#]+)")


def is_bad_sentence(s):
	if s == None:
		return True
	elif s == '\n':
		return True
	elif len(s) == 0:
		return True
	else:
		return False

def parse_sentence(sentence, vocab):
	# sentence to word_list
	word_list = []
	for m in MECAB.parse(sentence).split("\n"): # 単語に分解する
		w = m.split("\t")[0].lower() # 単語
		# stop wordsによる無駄な文字の排除
		if (w not in STOP_WORDS) and (w != "eos"):
			word_list.append(w)

	# Stop wordsの影響で違反文になった場合終了
	if is_bad_sentence("".join(word_list)):
		return None, vocab, False

	# 単語辞書の生成
	for word in word_list:
		if word not in vocab:
			vocab.append(word)

	# 単語辞書から，IDに変換
	ids = []
	for word in word_list:
		ids.append(str(vocab.index(word)))
	ids = ",".join(ids) + "\n"

	return ids, vocab, True

def find_data(soup, tag, class_=None):
	datas = soup.find_all(tag, class_=class_)
	datas = list(map(lambda s: s.string, datas))
	# 時間での昇順にする(fbのみ必要)
	datas.reverse()
	return datas

def write2file(usrs, msgs, que_file, res_file, queid_file, resid_file, vocab_file):
	# 偶数でなければ，最後の要素を削除(最後は'ok'などの返事不要なものであると仮定)
	if len(usrs) % 2 != 0:
		del usrs[-1:]
	if len(msgs) % 2 != 0:
		del msgs[-1:]

	switch = True
	before_usr = ""
	vocab = ["<eos>", "<unk>"]
	fq = open(que_file, "a")
	fr = open(res_file, "a")
	fqid = open(queid_file, "a")
	frid = open(resid_file, "a")
	for usr, msg in zip(usrs, msgs):
		# 同一人物の連続した発話は除外
		if (before_usr != usr) and not(is_bad_sentence(msg)):
			# URLの排除
			msg = RE_URL.sub("", msg)
			# もし無駄な改行があれば，排除して付ける
			msg = msg.replace("\n", "") + "\n"
			if not(is_bad_sentence(msg)): # もう一度チェック
				ids, vocab, done = parse_sentence(msg, vocab)
				if done: # さらにチェック
					fq.write(msg) if switch else fr.write(msg)
					fqid.write(ids) if switch else frid.write(ids)
					before_usr = usr
					switch = not(switch)
	fq.close()
	fr.close()
	fqid.close()
	frid.close()

	# 単語辞書の生成
	# 配列のindexがID
	with open(vocab_file, 'w') as f:
		for v in vocab:
			f.write(v + '\n')

def parse_fb(in_file, que_file, res_file, queid_file, resid_file, vocab_file):
	with open(in_file, "r") as f:
		soup = BeautifulSoup(f.read(), "html.parser")

	usrs = find_data(soup, "span", "user")
	msgs = find_data(soup, "p")

	write2file(usrs, msgs, que_file, res_file, queid_file, resid_file, vocab_file)

def parse_line(in_file, que_file, res_file, queid_file, resid_file, vocab_file):
	with open(in_file, "r") as f:
		lines = f.readlines()

	usrs = []
	msgs = []
	for line in lines[4:]: # 最初の４行は無視
		if RE_TIME.match(line):
			line_list = RE_TAB.split(line)
			if not RE_IGNORE.match(line_list[2]):
				usrs.append(line_list[1])
				msgs.append(line_list[2].strip().lstrip('"'))

	write2file(usrs, msgs, que_file, res_file, queid_file, resid_file, vocab_file)

def parse_corpus(in_file, que_file, res_file, queid_file, resid_file, vocab_file):
	with open(in_file, "r") as f:
		lines = f.readlines()

	re_kakko = re.compile(r"（.+）")
	re_keykakko = re.compile(r"【.+】")

	usrs = []
	msgs = []
	for i, line in enumerate(lines):
		# 最初の文はinputである前提で処理を行う
		if i % 2 == 0: # input
			usrs.append("input")
			msg = line.replace("input: ", "")
		else:          # output
			usrs.append("output")
			msg = line.replace("output: ", "")
		msg = msg.replace("＊＊＊", "")
		msg = msg.replace("＜笑い＞", "笑")
		msg = msg.replace("＜間＞", "")
		msg = msg.replace("\n", "")
		msg = re_kakko.sub("", msg)
		msg = re_keykakko.sub("", msg)
		msgs.append(msg)

	write2file(usrs, msgs, que_file, res_file, queid_file, resid_file, vocab_file)

def main():
	out_dir = "./data/"
	que_file = out_dir + "query.txt"
	res_file = out_dir + "response.txt"
	queid_file = out_dir + "query_id.txt"
	resid_file = out_dir + "response_id.txt"
	vocab_file = out_dir + "vocab.txt"
	try:
		os.mkdir(out_dir)
	except: # ディレクトリが存在する時
		try:
			# 追記モードなので事前に削除しておく
			os.remove(que_file)
			os.remove(res_file)
			os.remove(queid_file)
			os.remove(resid_file)
		except:
			pass

	fb_dir = "./raw/facebook/messages/"
	fb_files = list(map(lambda s: fb_dir + s, os.listdir(fb_dir)))
	for fb in fb_files:
		parse_fb(fb, que_file, res_file, queid_file, resid_file, vocab_file)

	line_dir = "./raw/line/"
	listdir = os.listdir(line_dir)
	listdir.remove(".gitkeep")
	line_files = list(map(lambda s: line_dir + s, listdir))
	for line in line_files:
		parse_line(line, que_file, res_file, queid_file, resid_file, vocab_file)

	corpus = "./raw/corpus/sequence.txt"
	parse_corpus(corpus, que_file, res_file, queid_file, resid_file, vocab_file)

	print("done.")

if __name__ == '__main__':
	main()