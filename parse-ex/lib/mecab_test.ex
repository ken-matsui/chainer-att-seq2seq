defmodule MecabTest do
    @moduledoc """
    Documentation for MecabTest.
    """

    @doc """
    Hello world.

    ## Examples

            iex> MecabTest.hello
            :world

    """
    def mecab(line) do
        Mecab.parse(line, mecab_option: "-d /usr/local/lib/mecab/dic/mecab-ipadic-neologd")
        |> Enum.reject(&(&1 == nil))
        |> Enum.map(&(Map.get(&1, "surface_form")))
    end

    def stop_words(wordlist, [stopword | _]) when stopword == "" do
        List.delete(wordlist, stopword)
    end
    def stop_words(wordlist, [stopword | tail]) do
        List.delete(wordlist, stopword)
        |> stop_words(tail)
    end

    # Base
    def compress([head | tail], beforeuser, message, talk) when tail == [] do
        nowuser = head |> List.first()
        #     t - 1   ==    t    ?
        if beforeuser == nowuser do
            message = message <> "EOS" <> (head |> Enum.at(1))
            talk ++ [message]
        else
            talk ++ [message] ++ [head |> Enum.at(1)]
        end
        # :return: talk
    end
    # Recursion
    def compress([head | tail], beforeuser, message, talk) do
        nowuser = head |> List.first()
        #     t - 1   ==    t    ?
        if beforeuser == nowuser do
            message = message <> "EOS" <> (head |> Enum.at(1))
            compress(tail, nowuser, message, talk)
        else # Userが切り替わったら，talkにpushする．
            compress(tail, nowuser, (head |> Enum.at(1)), talk ++ [message])
        end
    end
    # Initial
    def compress(lines) do
        # [["user", "message"], ...] -> ["user", "msg"] -> "user"
        firstuser = lines |> List.first() |> List.first()
        # At first is query
        message = lines |> List.first() |> Enum.at(1)
        # [["user1", "msg1"], ["user1", "msg2"], ["user2", "msg3"], ...]
        #      -> ["msg1EOSmsg2", "msg3", ...]
        compress(List.delete_at(lines, 0), firstuser, message, [])
    end

    # "raw/line/"
    def parse_line(dir) do
        stopword_url = "http://svn.sourceforge.jp/svnroot/slothlib/CSharp/Version1/SlothLib/NLP/Filter/StopWord/word/Japanese.txt"
        dlpath = "raw/stopwords.txt"
        case Download.from(stopword_url, [path: dlpath]) do
            {:ok, _} -> :ok
            {:error, :eexist} -> :exist
            {:error, err} -> throw err
        end
        stopwords = (File.read!(dlpath) |> String.split("\r\n"))

        {:ok, time_pt} = Regex.compile("[0-9]?[0-9]:[0-9]?[0-9]\t")
        {:ok, qurt_pt} = Regex.compile("\"")
        {:ok, ignr_pt} = Regex.compile("(\\[Photo\\]|\\[Sticker\\]|\\[Video\\]|\\[Albums\\]|\\[File\\]|☎)")
        {:ok, urls_pt} = Regex.compile("(https?|ftp)(://[-_\\.!~*'()a-zA-Z0-9;/?:@&=\\+\\$,%\\#]+)")

        # "raw/line/" -> ["example1.txt", "...txt", ...]
        File.ls!(dir)
        # ["example1.txt", ".DS_Store", ...] -> ["example1.txt", ...]
        |> List.delete(".DS_Store")
        # ["example1.txt", ...] -> ["raw/line/example1.txt", ...]
        |> Stream.map(&(dir <> &1))
        # ["raw/line/example1.txt", ...] -> ["text\r\ntext\r\n..."]
        |> Stream.map(&(Task.async(fn -> File.read!(&1) end)))
        |> Stream.map(&Task.await/1)
        # ["text\r\ntext\r\n..."] -> [["text", "text", ...]]
        |> Enum.map(&(String.split(&1, "\r\n")))
        # [["text", "text", ...]] -> ["text", "text", ...]
        |> List.flatten
        # ["text", "time\tuser\tmsg", ...] -> ["time\tuser\tmsg", ...]
        |> Stream.filter(&(Regex.match?(time_pt, &1)))
        # ["time\tuser\tmsg", ...] -> ["user\tmsg", ...]
        |> Stream.map(&(Regex.replace(time_pt, &1, "")))
        # ["user\t'msg'", ...] -> ["user\tmsg", ...]
        |> Stream.map(&(Regex.replace(qurt_pt, &1, "")))
        # ["user\t[Sticker]", "user\tmsg", ...] -> ["user\tmsg", ...]
        |> Stream.reject(&(Regex.match?(ignr_pt, &1)))
        # ["user\thttp://~~~", "user\tmsg", ...] -> ["user\tmsg", ...]
        |> Stream.reject(&(Regex.match?(urls_pt, &1)))
        # ["user\tmsg", ...] -> [["user", "msg"], ...]
        |> Enum.map(&(String.split(&1, "\t")))
        # [["user", "msg"], ...] -> ["query", "response", "query", ...]
        |> compress()
        # ["query", "response", ...] -> [["word1", "word2", ...], [...], ...]
        |> Stream.map(&(Task.async(fn -> mecab(&1) end)))
        |> Stream.map(&Task.await/1)
        # Delete "EOS"
        |> Enum.map(&(Enum.reject(&1, fn(s) -> s == "EOS" end)))
        # Delete word included in stopwords
        # |> Stream.map(&(stop_words(&1, stopwords)))
        # [["w1", "w2", "w3"], [...], ...] -> ["w1w2w3", ...]
        # |> Enum.map(&(Enum.join(&1, "")))
    end
    # :timer.tc(fn -> MecabTest.parse_line("raw/line/") end)

    def parse_all() do
        # If odd, convert it to even.
        line = parse_line("raw/line/")
        if line |> Enum.count() |> rem(2) != 0 do
            line = List.delete_at(line, -1)
        end
        # parse_line() ++ parse_fb())
        # # ファイルへ書き込み(Taskで？)
        # |>
    end
end
