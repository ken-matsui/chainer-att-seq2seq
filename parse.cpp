// Inside
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <memory>
#include <algorithm>
#include <chrono>
#include <iterator>
// #include <string_view>

// Outside
#include <mecab.h>
#include <boost/range.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/xpressive/xpressive.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/range/adaptor/indexed.hpp>
#include <boost/algorithm/string/classification.hpp> // is_space()

// Made me
#include <htmlparser.hpp>

// Debug(Inside)
#include <typeinfo> // typeid( obj ).name()


void make_data(std::vector<std::string>& word_list,
			   std::string& dialogue,
			   std::string& ids,
			   std::vector<std::string>& vocab)
{
	// Make unique vocabulary, and Make list word to ID.
	for (const std::string& word : word_list) {
		if(!word.empty()) {
			auto itr = std::find(vocab.begin(), vocab.end(), word);
			std::size_t index = std::distance(vocab.begin(), itr);
			if(index != vocab.size()) { // vocabに既に含まれている時
				ids += std::to_string(index) + ",";
			}
			else { // 未登録の時
				ids += std::to_string(vocab.size()) + ",";
				vocab.emplace_back(word);
			}
			dialogue += word + ",";
		}
	}
}

void split_mecab(const std::unique_ptr<MeCab::Tagger>&& tagger,
				std::string& input,
				std::vector<std::string>& output)
{
	// tagger->parse is <const char *>
	const std::string&& result = tagger->parse( input.data() );
	boost::algorithm::split(output, result, boost::is_space());
}

void readlines(std::vector<std::string>& lines, const boost::filesystem::path&& path)
{
	namespace fs = boost::filesystem;
	for (const auto& e : boost::make_iterator_range(fs::directory_iterator(path), {})) {
		if (!fs::is_directory( e )) {
			std::ifstream ifs( e.path().string() );
			std::string line;
			while (getline(ifs, line))
				if (line.size() != 1)
					lines.emplace_back(line);
		}
	}
}
void readlines(std::string& lines, const boost::filesystem::path&& path)
{
	namespace fs = boost::filesystem;
	for (const auto& e : boost::make_iterator_range(fs::directory_iterator(path), {})) {
		if (!fs::is_directory( e )) {
			std::ifstream ifs( e.path().string() );
			std::string line;
			while (getline(ifs, line))
				if (line.size() != 1)
					lines += line;
		}
	}
}

void parse_line(const boost::filesystem::path&& path,
				std::vector<std::vector<std::string>>& datas,
				std::vector<std::string>& vocab)
{
	namespace regex = boost::xpressive;
	// (time= [0-9][0-9]:[0-9][0-9]\t)(user= ((?:.)+))\t(msg= ((?:.)+))
	regex::mark_tag time(1), user(2), msg(3);
	regex::smatch match;
	const regex::sregex re_time = (time= regex::range('0','9') >> regex::range('0','9') >> ":"
						 >> regex::range('0','9') >> regex::range('0','9') >> "\t")
						 >> (user= +regex::_)
						 >> "\t"
						 >> (msg= +regex::_);
	const regex::sregex re_ignr = regex::keep("[Photo]")
						  | regex::keep("[Sticker]")
						  | regex::keep("[Video]")
						  | regex::keep("[Albums]")
						  | regex::keep("[File]")
						  | regex::keep("☎");
	const regex::sregex re_urls = !regex::keep("https")
						 >> regex::keep("://")
						 >> +regex::_;

	const std::unique_ptr<MeCab::Tagger> tagger{ MeCab::createTagger("-Owakati") };

	std::vector<std::string> lines;
	readlines(lines, std::move(path));

	std::string dialogue;
	std::string ids;
	std::string before_user{ "" };
	for (const std::string& line : lines) {
		if(!regex::regex_search(line, re_ignr) &&
		   !regex::regex_search(line, re_urls) &&
			regex::regex_search(line, match, re_time))
		{
			std::string match_user( match[user] );
			std::string match_msg( match[msg] );
			if (before_user != "" && before_user != match_user) {
				datas.emplace_back(std::vector<std::string>{ before_user, dialogue, ids });
				ids = "";
				dialogue = "";
			}
			before_user = match_user;
			std::vector<std::string> word_list;
			split_mecab(std::move(tagger), match_msg, word_list);
			make_data(word_list, dialogue, ids, vocab);
		}
	}
	// 最後はどうせ，ok的な，返信不要なもの．と仮定する．
	// datas.emplace_back(std::vector<std::string>{ before_user, dialogue, ids });
}

void parse_fb(const boost::filesystem::path&& path,
			std::vector<std::vector<std::string>>& datas,
			std::vector<std::string>& vocab)
{
	namespace regex = boost::xpressive;
	const regex::sregex re_urls = !regex::keep("https")
						 >> regex::keep("://")
						 >> +regex::_;

	const std::unique_ptr<MeCab::Tagger> tagger{ MeCab::createTagger("-Owakati") };

	std::string lines;
	readlines(lines, std::move(path));

	std::vector<std::string> users;
	htmlparser::find_all(lines.begin(), lines.end(), users, "span", "user");
	std::reverse(users.begin(), users.end());
	std::vector<std::string> msgs;
	htmlparser::find_all(lines.begin(), lines.end(), msgs, "p");
	std::reverse(msgs.begin(), msgs.end());

	std::string dialogue;
	std::string ids;
	std::string before_user{ "" };
	for (const auto& msg : msgs | boost::adaptors::indexed()) {
		if(!regex::regex_search(msg.value(), re_urls) && msg.value() != "") {
			if (before_user != "" && before_user != users[msg.index()]) {
				datas.emplace_back(std::vector<std::string>{ before_user, dialogue, ids });
				ids = "";
				dialogue = "";
			}
			before_user = users[msg.index()];
			std::vector<std::string> word_list;
			split_mecab(std::move(tagger), msg.value(), word_list);
			make_data(word_list, dialogue, ids, vocab);
		}
	}
	// 最後はどうせ，ok的な，返信不要なもの．と仮定する．
	// datas.emplace_back(std::vector<std::string>{ before_user, dialogue, ids });
}

void write2files(const boost::filesystem::path path,
				std::vector<std::vector<std::string>>& datas,
				std::vector<std::string>& vocab)
{
	boost::filesystem::create_directory(path);
	std::ofstream data_ofs(path.string() + "data.txt");
	std::ofstream dataid_ofs(path.string() + "dataid.txt");
	for(const auto& d : datas | boost::adaptors::indexed()) {
		if(d.index() % 2 == 0) {
			data_ofs << d.value()[0] << ":" << d.value()[1];
			dataid_ofs << d.value()[2];
		}
		else {
			data_ofs << "\t" << d.value()[0] << ":" << d.value()[1] << std::endl;
			dataid_ofs << "\t" << d.value()[2] << std::endl;
		}
	}
	std::ofstream vocab_ofs(path.string() + "vocab.txt");
	for(std::string v : vocab)
		vocab_ofs << v << std::endl;
}

int main(int argc, char **argv) {
	auto start = std::chrono::system_clock::now();

	// vector[?] -> vector[0]="user"[1]="id"[2]="msg" -> string
	std::vector<std::vector<std::string>> datas;
	std::vector<std::string> vocab{ "<eos>", "<unk>" };

	std::cout << "Start parse of line..." << std::endl;
	parse_line(boost::filesystem::path("raw/line/"), datas, vocab);
	std::cout << "End parse of line." << std::endl << std::endl;

	std::cout << "Start parse of facebook..." << std::endl;
	parse_fb(boost::filesystem::path("raw/facebook/messages/"), datas, vocab);
	std::cout << "End parse of facebook." << std::endl << std::endl;


	write2files(boost::filesystem::path("data/"), datas, vocab);
	std::cout << "done." << std::endl;

	auto end = std::chrono::system_clock::now();
	auto msec = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	std::cout << "It took " << msec << "ms." << std::endl;
}