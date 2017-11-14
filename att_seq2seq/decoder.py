from chainer import serializers

class Decoder(object):
	def __init__(self, model, data_converter, npz):
		self.model = model
		self.data_converter = data_converter
		serializers.load_npz(npz, self.model)

	def __call__(self, query):
		enc_query = self.data_converter.sentence2ids(query, train=False)
		dec_response = self.model(enc_words=enc_query, train=False)
		response = self.data_converter.ids2words(dec_response)
		if "<eos>" in response: # 最後の<eos>を回避
			res = "".join(response[0:-1])
		else: # 含んでない時もある．(出力wordサイズが，15を超えた時？？？)
			res = "".join(response)
		return res
