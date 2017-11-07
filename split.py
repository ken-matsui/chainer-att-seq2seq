# coding:utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
from tqdm import tqdm

def main():
	# 出力ファイルのディレクトリ生成
	out_dir = "./datas/"
	if not os.path.exists(out_dir):
		os.mkdir(out_dir)

	# mecabデータの読み込み
	msg = open("./mecab_out.txt", "r")
	lines = msg.readlines()
	msg.close()

	# 1つ飛ばしでループして，交互になってるinput&outputを別ファイルへ分割
	input = open(out_dir + "input.txt", "w")
	output = open(out_dir + "output.txt", "w")
	for i in tqdm(range(0, len(lines), 2)):
		input.write(lines[i])
		output.write(lines[i + 1])

if __name__ == '__main__':
	main()