// libxmlは，Cで書かれてて，https://github.com/lexborisov/myhtml は，なんか使い方が一目でわからない時点でクソだし，ヘッダーオンリーじゃないし，，，，

#ifndef __HTMLPARSER_HPP__
#define __HTMLPARSER_HPP__

#include <string>
#include <vector>
#include <iostream>

namespace htmlparser
{
	template <class InputIterator, class RangeIterator>
	InputIterator search_tag(InputIterator first1, InputIterator last1, RangeIterator first2, RangeIterator last2)
	{
		RangeIterator tag_itr = first2;
		for ( ; first1 != last1; ++first1)
		{
			if (*first1 == *tag_itr)
			{
				for (InputIterator itr = first1;
					itr != last1 && tag_itr != last2 && *itr == *tag_itr;
					++itr, ++tag_itr);
				if (tag_itr == last2) return first1;
				else tag_itr = first2;
			}
		}
		return last1;
	}

	template <class InputIterator, class RangeIterator>
	InputIterator find_tag(InputIterator first1, InputIterator last1, RangeIterator first2, RangeIterator last2)
	{
		RangeIterator tag_itr = first2;
		for ( ; first1 != last1; ++first1)
		{
			if (*first1 == *tag_itr)
			{
				for (InputIterator itr = first1; itr != last1 && tag_itr != last2; ++itr)
				{
					if (*itr == *tag_itr) {
						++tag_itr;
					}
				}
				if (tag_itr == last2) return first1;
				else tag_itr = first2;
			}
		}
		return last1;
	}

	template <class InputIterator, class stringT, class charT>
	bool find(InputIterator first, InputIterator last, stringT& out, const charT& tag)
	{
		const std::string begin_tag = "<" + static_cast<std::string>(tag) + ">";
		InputIterator begin_tag_itr = search_tag(first, last, begin_tag.begin(), begin_tag.end());
		if (begin_tag_itr == last) return false;

		const std::string end_tag = "</" + static_cast<std::string>(tag) + ">";
		InputIterator end_tag_itr = search_tag(begin_tag_itr+begin_tag.size(), last, end_tag.begin(), end_tag.end());
		if (end_tag_itr == last) return false;

		std::copy(begin_tag_itr+begin_tag.size(), end_tag_itr, std::back_inserter(out));
		return true;
	}

	template <class InputIterator, class vsT>
	bool find_all(InputIterator first, InputIterator last, vsT& out, const std::string& tag, const std::string& class_="NULL")
	{
		std::string begin_tag;
		if (class_ == "NULL")
			begin_tag = "<" + static_cast<std::string>(tag) + ">";
		else
			begin_tag = "<" + static_cast<std::string>(tag) + " class=\"" + class_ + "\">";
		const std::string end_tag   = "</"+ static_cast<std::string>(tag) + ">";

		while (true)
		{
			InputIterator begin_tag_itr = search_tag(first, last, begin_tag.begin(), begin_tag.end());
			if (begin_tag_itr == last) break;
			InputIterator end_tag_itr = search_tag(begin_tag_itr+begin_tag.size(), last, end_tag.begin(), end_tag.end());
			if (end_tag_itr == last)   break;

			first = end_tag_itr;

			std::string result;
			std::copy(begin_tag_itr+begin_tag.size(), end_tag_itr, std::back_inserter(result));
			out.push_back(result);
		}

		if (out.size() == 0) return false;
		else                 return true;
	}
}

#endif
