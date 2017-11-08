# coding:utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import MeCab
from tqdm import tqdm

def main():
	# 辞書を指定
	mecab = MeCab.Tagger('-d /usr/local/Cellar/mecab-ipadic/2.7.0-20070801/lib/mecab/dic/ipadic')

	# messageディレクトリのリスト生成
	in_dir = "./messages/"
	files = os.listdir(in_dir)
	# 出力ファイルのディレクトリ生成
	out_dir = "./datas/"
	if not os.path.exists(out_dir):
		os.mkdir(out_dir)

	# patternのcompile
	ptrn_msg = re.compile(r"msg: ")
	ptrn_usr = re.compile(r"user: ")

	with open(out_dir + "input.txt", "w") as fin, open(out_dir + "output.txt", "w") as fout:
		switch = True # True => fin, False => fout
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
						fin.write(word + " ") if switch else fout.write(word + " ")
						node = node.next # 次の単語に進める
				elif ptrn_usr.match(line):
					if not user: # 初回ループ
						user = line
					elif user != line: # 前回userと違う
						user = line
						fin.write("\n") if switch else fout.write("\n")
						# input と outputを切り替え
						switch = not(switch)
						user = ""

if __name__ == '__main__':
	main()