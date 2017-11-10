# coding:utf-8

# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
# from __future__ import unicode_literals

import os
import re
import codecs
from bs4 import BeautifulSoup
from tqdm import tqdm

def main():
	# htmlファイルのリスト生成
	fb_dir = "./raw/facebook/messages/"
	files = map(lambda s: fb_dir + s, os.listdir(fb_dir))
	line_dir = "./raw/line/"
	files += map(lambda s: line_dir + s, os.listdir(line_dir))
	# .gitkeepを排除
	files.remove(line_dir + ".gitkeep")

	# 出力ファイルのディレクトリ生成
	out_dir = "./middle/"
	if not os.path.exists(out_dir):
		os.mkdir(out_dir)

	# patternのcompile
	re_fb = re.compile(r"%s" %fb_dir)
	re_line = re.compile(r"%s" %line_dir)
	re_time = re.compile(r"[0-9]?[0-9]:[0-9]?[0-9]")
	# 返信の際に，スタンプなら[Sticker]に文字列変換して，seq2seqに渡せば返信可能になるかも．
	re_ignore = re.compile(r"(\[Photo\]|\[Sticker\]|\[Video\]|\[Albums\]|\[File\])")
	re_tab = re.compile(r"\t")
	re_space = re.compile(r" ")

	for file in tqdm(files):
		with codecs.open(file, "r", "utf-8") as f:
			users = []
			messages = []
			if re_fb.match(file):
				soup = BeautifulSoup(f.read(), "html.parser")
				partner = soup.find("title").string.replace("スレッドの相手: ", "").replace(" ", "")
				users = soup.find_all("span", class_="user")
				users = map(lambda s: s.string, users)
				users.reverse()
				messages = soup.find_all("p")
				messages = map(lambda s: s.string, messages)
				messages.reverse()
				ext = ".fb"
			elif re_line.match(file):
				lines = f.readlines()
				partner = lines[0].replace("[LINE] Chat history with ", "").replace(" ", "").strip()
				for line in lines[4:]: # 最初の４行は無視
					if re_time.match(line):
						line_list = re_tab.split(line)
						if not re_ignore.match(line_list[2]):
							users.append(line_list[1])
							messages.append(line_list[2].strip().lstrip('"'))
				ext = ".line"

			# トーク相手名をファイル名にする
			f = codecs.open(out_dir + partner + ext, "w", "utf-8")
			for user, msg in zip(users, messages):
				if msg is not None:
					f.write("user: " + user + "\n")
					f.write("msg: " + msg + "\n")
					f.write("\n")
			f.close()

if __name__ == '__main__':
	main()