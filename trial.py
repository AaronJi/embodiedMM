import os
from fairseq.data import Dictionary, encoders
import torch
from omegaconf import DictConfig

bpe_dir='utils/BPE'

src_dict = Dictionary.load(os.path.join(bpe_dir, "dict.txt"))
tgt_dict = Dictionary.load(os.path.join(bpe_dir, "dict.txt"))
#print(src_dict)
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

'''
e = src_dict.encode_line('I want a boy')
print(e)
i = src_dict.index('I')
print(i)
t = torch.tensor([50264, 50265, 50266, 50267,     2], dtype=torch.int32)
s = src_dict.string(t)
print(s)
'''

bpe_dict = {
    "_name": "gpt2",
    "gpt2_encoder_json": os.path.join(bpe_dir, "encoder.json"),
    "gpt2_vocab_bpe": os.path.join(bpe_dir, "vocab.bpe")
}
bpe_dict = DictConfig(bpe_dict)
bpe = encoders.build_bpe(bpe_dict)
#print(type(bpe))
#be = bpe.encode('I want a boy')




text = " what does the image describe?"
be = bpe.encode(text)
print(be)
print(type(be))
bd = bpe.decode(be)
print(bd)
#tde = tgt_dict.encode_line(line=bd, add_if_not_exist=False, append_eos=False).long()
#rint(tde)
use_bpe = True
tde = tgt_dict.encode_line(line=bpe.encode(text) if use_bpe else text, add_if_not_exist=False, append_eos=False).long()
print(tde)


#s = self.tgt_dict.encode_line(line=self.bpe.encode(text) if use_bpe else text, add_if_not_exist=False, append_eos=False).long()
from data.file_dataset import FileDataset
file_path = 'dataset/pretrain_data/vision_language_examples.tsv'
selected_cols='0,1,2,3,4,5,6,7'
dataset = FileDataset(file_path, selected_cols)
print(dataset)
index = 0
uniq_id, image, caption, question, refs, gt_objects, dataset_name, type0 = dataset[index]
print(uniq_id)
#print(image)
print(type(image))
print(caption)
#print(question)
#print(refs)
#print(gt_objects)
print(dataset_name)
print(type0)

