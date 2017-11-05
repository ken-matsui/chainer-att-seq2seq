# coding:utf-8

import os
import re
import MeCab
from tqdm import tqdm

def main():
	# 辞書を指定
	mecab = MeCab.Tagger('-d /usr/local/Cellar/mecab-ipadic/2.7.0-20070801/lib/mecab/dic/ipadic')

	# messageファイルのリスト生成
	in_dir = "./messages/"
	files = os.listdir(in_dir)

	# patternのcompile
	pattern = r"msg: "
	ptrn_msg = re.compile(pattern)
	pattern = r"user: "
	ptrn_usr = re.compile(pattern)

	f = open("./mecab_out.txt", "w")
	for file in tqdm(files):
		msg = open(in_dir + file, "r")
		lines = msg.readlines()
		msg.close()

		user = ""
		for line in lines:
			if not line: # 空行
				continue
			if ptrn_msg.match(line):
				text = line.replace("msg: ", "")
				mecab.parse('') # 文字列がGCされるのを防ぐ
				node = mecab.parseToNode(text)
				while node:
					word = node.surface # 単語を取得
					f.write(word + " ")
					node = node.next # 次の単語に進める
			elif ptrn_usr.match(line):
				if not user: # 初回ループ
					user = line
				elif user != line: # 前回userと違う
					user = line
					f.write("\n")
		f.write("\n")
	f.close()

if __name__ == '__main__':
	main()