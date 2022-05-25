import os
from fairseq.data import Dictionary, encoders
import torch
from omegaconf import DictConfig

bpe_dir='utils/BPE'

src_dict = Dictionary.load(os.path.join(bpe_dir, "dict.txt"))
tgt_dict = Dictionary.load(os.path.join(bpe_dir, "dict.txt"))
print(src_dict)

e = src_dict.encode_line('I want a boy')
print(e)

i = src_dict.index('I')
print(i)

t = torch.tensor([50264, 50265, 50266, 50267,     2], dtype=torch.int32)
s = src_dict.string(t)
print(s)

code_dict_size = 8192
num_bins = 1000

src_dict.add_symbol("<mask>")
tgt_dict.add_symbol("<mask>")
for i in range(code_dict_size):
    src_dict.add_symbol("<code_{}>".format(i))
    tgt_dict.add_symbol("<code_{}>".format(i))
# quantization
for i in range(num_bins):
    #print("<bin_{}>".format(i))
    src_dict.add_symbol("<bin_{}>".format(i))
    tgt_dict.add_symbol("<bin_{}>".format(i))

print(len(src_dict))
#print(src_dict.index("<bin_{4}>"))
for i in range(num_bins):
    ii = src_dict.index("<bin_{}>".format(i))
    #print(ii)

bpe_dict = {
    "_name": "gpt2",
    "gpt2_encoder_json": os.path.join(bpe_dir, "encoder.json"),
    "gpt2_vocab_bpe": os.path.join(bpe_dir, "vocab.bpe")
}
bpe_dict = DictConfig(bpe_dict)
bpe = encoders.build_bpe(bpe_dict)
print(type(bpe))
be = bpe.encode('I want a boy')
print(be)
print(type(be))
bd = bpe.decode('40 765 257 2933')
print(bd)
