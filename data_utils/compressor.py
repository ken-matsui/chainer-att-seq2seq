# coding:utf-8

import os
import re

def main():
	'''
	middleデータをinput outputの二つにする．
	'''
	# messageディレクトリのリスト生成
	in_dir = "./middle/"
	files = os.listdir(in_dir)
	# 出力ファイルのディレクトリ生成
	out_dir = "./data/"
	if not os.path.exists(out_dir):
		os.mkdir(out_dir)

	# patternのcompile
	ptrn_usr = re.compile(r"user: ")
	ptrn_msg = re.compile(r"msg: ")

	with open(out_dir + "input.txt", "w") as fin, open(out_dir + "output.txt", "w") as fout:
		switch = True # True => fin, False => fout
		user = ""
		for file in files:
			with open(in_dir + file, "r") as msg:
				lines = msg.readlines()
			for line in lines:
				if not line: # 空行
					continue
				if ptrn_usr.match(line):
					if not user: # 初回ループ
						user = line
					elif user != line: # 前回userと違う
						user = line
						fin.write("\n") if switch else fout.write("\n")
						# input と outputを切り替え
						switch = not(switch)
				elif ptrn_msg.match(line):
					text = line.replace("msg: ", "")
					fin.write(text.strip()) if switch else fout.write(text.strip())

	print("done.")

if __name__ == '__main__':
	main()