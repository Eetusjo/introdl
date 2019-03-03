import os, re
from collections import defaultdict as dd
import nltk
import torch

WORD_BOUNDARY='#'
UNK='UNK'

def read_lines(fn):
    data = []
    for line in open(fn):
        line = line.strip('\n')
        if line:
            src = " ".join(str(line.split("\t")[0])) + " " +re.sub(",", " ", line.split("\t")[1])
            src = re.sub("   ", " <SPACE> ", src)
            tgt = " ".join(line.split("\t")[2])
            tgt = re.sub("   ", " <SPACE> ", tgt)
            line = src+' '+tgt
            data.append({'SOURCE': src,
                         'TARGET': tgt,
                         'TOKENIZED_SOURCE': src.split(),
                         'TOKENIZED_TARGET': tgt.split(),
                         'TOKENIZED_LINE': line.split()})

    return data


def compute_tensor(word_ex,charmap):
    word_ex['SOURCE_TENSOR'] = torch.LongTensor([charmap[WORD_BOUNDARY]]
                                     + [charmap[c] if c in charmap 
                                                   else charmap[UNK] 
                                                   for c in word_ex['TOKENIZED_SOURCE']]
                                     + [charmap[WORD_BOUNDARY]])

    word_ex['TARGET_TENSOR'] = torch.LongTensor([charmap[WORD_BOUNDARY]]
                                     + [charmap[c] if c in charmap 
                                                   else charmap[UNK] 
                                                   for c in word_ex['TOKENIZED_TARGET']]
                                     + [charmap[WORD_BOUNDARY]])


def read_datasets(prefix,data_dir):
    datasets = {'training': read_lines(os.path.join(data_dir, '%s-%s' % 
                                                    (prefix, 'train'))), 
                'dev': read_lines(os.path.join(data_dir, '%s-%s' % 
                                               (prefix, 'dev'))),
                'test': read_lines(os.path.join(data_dir, '%s-%s' %
                                                (prefix, 'test')))}


    charmap = {c:i for i,c in enumerate({c for ex in datasets['training'] 
                                         for c in ex['TOKENIZED_LINE']})}
    charmap[UNK] = len(charmap)
    charmap[WORD_BOUNDARY] = len(charmap)

    for word_ex in datasets['training']:
        compute_tensor(word_ex,charmap)
    for word_ex in datasets['dev']:
        compute_tensor(word_ex,charmap)
    for word_ex in datasets['test']:
        compute_tensor(word_ex,charmap)

    return datasets, charmap


if __name__=='__main__':
    from paths import data_dir



