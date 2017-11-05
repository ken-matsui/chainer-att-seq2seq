# coding:utf-8

import os
from bs4 import BeautifulSoup
from tqdm import tqdm

def main():
	# htmlファイルのリスト生成
	in_dir = "./facebook-matken11235/messages/"
	files = os.listdir(in_dir)
	# 出力ファイルのディレクトリ生成
	out_dir = "./messages/"
	if not os.path.exists(out_dir):
		os.mkdir(out_dir)

	for file in tqdm(files):
		html = open(in_dir + file, "r")
		soup = BeautifulSoup(html.read(), "html.parser")
		html.close()

		partner = soup.find("title")
		users = soup.find_all("span", class_="user")
		times = soup.find_all("span", class_="meta")
		messages = soup.find_all("p")

		# トーク相手名をファイル名にする
		filename = partner.string.encode('utf-8').replace("スレッドの相手: ", "").replace(" ", "")

		# ファイル作成
		f = open(out_dir + filename + ".txt", "w")
		# 時系列の昇順に並べ換える
		times.reverse()
		users.reverse()
		messages.reverse()
		for time, user, msg in zip(times, users, messages):
			if msg.string is not None:
				f.write("time: " + time.string.encode('utf-8') + "\n")
				f.write("user: " + user.string.encode('utf-8') + "\n")
				f.write("msg: " + msg.string.encode('utf-8') + "\n")
				f.write("\n")
		f.close()

if __name__ == '__main__':
	main()