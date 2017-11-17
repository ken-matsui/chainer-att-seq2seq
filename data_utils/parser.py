# coding:utf-8

import os
import re

from bs4 import BeautifulSoup

RE_TIME = re.compile(r"[0-9]?[0-9]:[0-9]?[0-9]")
RE_TAB = re.compile(r"\t")
RE_IGNORE = re.compile(r"(\[Photo\]|\[Sticker\]|\[Video\]|\[Albums\]|\[File\]|☎)")

def find_data(soup, tag, class_=None):
	datas = soup.find_all(tag, class_=class_)
	datas = list(map(lambda s: s.string, datas))
	datas.reverse() # 時間での昇順にする
	return datas

def write2file(usrs, msgs, que_file, res_file):
	# 偶数でなければ，最後の要素を削除(最後は'ok'などの返事不要なものであると仮定)
	if len(usrs) % 2 != 0:
		del usrs[-1:]
	if len(msgs) % 2 != 0:
		del msgs[-1:]

	before_usr = ""
	switch = True # True => fq, False => fr
	with open(que_file, "a") as fq, open(res_file, "a") as fr:
		for usr, msg in zip(usrs, msgs):
			# 同一人物の連続した発話は除外 && 空行も除外
			if (before_usr != usr) and (msg is not None) and (msg is not '\n'):
				msg = msg.replace("\n", "") + "\n"
				fq.write(msg) if switch else fr.write(msg)
				before_usr = usr
				switch = not(switch)

def parse_fb(in_file, que_file, res_file):
	with open(in_file, "r") as f:
		soup = BeautifulSoup(f.read(), "html.parser")

	usrs = find_data(soup, "span", "user")
	msgs = find_data(soup, "p")

	write2file(usrs, msgs, que_file, res_file)

def parse_line(in_file, que_file, res_file):
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

	write2file(usrs, msgs, que_file, res_file)

def parse_corpus(in_file, que_file, res_file):
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

	write2file(usrs, msgs, que_file, res_file)

def main():
	out_dir = "./data/"
	que_file = out_dir + "query.txt"
	res_file = out_dir + "response.txt"
	try:
		os.mkdir(out_dir)
	except: # ディレクトリが存在する時
		try:
			os.remove(que_file)
			os.remove(res_file)
		except:
			pass

	fb_dir = "./raw/facebook/messages/"
	fb_files = list(map(lambda s: fb_dir + s, os.listdir(fb_dir)))
	for fb in fb_files:
		parse_fb(fb, que_file, res_file)

	line_dir = "./raw/line/"
	listdir = os.listdir(line_dir)
	listdir.remove(".gitkeep")
	line_files = list(map(lambda s: line_dir + s, listdir))
	for line in line_files:
		parse_line(line, que_file, res_file)

	corpus = "./raw/corpus/sequence.txt"
	parse_corpus(corpus, que_file, res_file)

	print("done.")

if __name__ == '__main__':
	main()