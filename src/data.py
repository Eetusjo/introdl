import nltk
import numpy as np
import re
import os
import torch

from collections import defaultdict as dd

WORD_START = "<w>"
WORD_END = "</w>"
PADDING = '#'
UNK = 'UNK'

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
    word_ex['SOURCE_TENSOR'] = torch.LongTensor([charmap[WORD_START]]
                                     + [charmap[c] if c in charmap
                                                   else charmap[UNK]
                                                   for c in word_ex['TOKENIZED_SOURCE']]
                                     + [charmap[WORD_END]])

    word_ex['TARGET_TENSOR'] = torch.LongTensor(
        [charmap[WORD_START]] + [charmap[c] if c in charmap else charmap[UNK]
                                 for c in word_ex['TOKENIZED_TARGET']]
        + [charmap[WORD_END]])


def read_datasets(prefix,data_dir):
    datasets = {'training': read_lines(os.path.join(data_dir, '%s-%s' %
                                                    (prefix, 'train'))),
                'dev': read_lines(os.path.join(data_dir, '%s-%s' %
                                               (prefix, 'dev'))),
                'test': read_lines(os.path.join(data_dir, '%s-%s' %
                                                (prefix, 'test')))}

    charmap = {c: i for i, c in enumerate({c for ex in datasets['training']
                                          for c in ex['TOKENIZED_LINE']})}
    charmap[UNK] = len(charmap)
    charmap[WORD_START] = len(charmap)
    charmap[WORD_END] = len(charmap)
    charmap[PADDING] = len(charmap)

    for word_ex in datasets['training']:
        compute_tensor(word_ex, charmap)
    for word_ex in datasets['dev']:
        compute_tensor(word_ex, charmap)
    for word_ex in datasets['test']:
        compute_tensor(word_ex, charmap)

    return datasets, charmap


def get_minibatch(minibatchwords, character_map, languages):
    src_word_lengths = [len(x['SOURCE_TENSOR']) for x in minibatchwords]
    tgt_word_lengths = [len(x['TARGET_TENSOR']) for x in minibatchwords]

    mb_src, mb_tgt = pad_minibatch(minibatchwords, src_word_lengths,
                                   tgt_word_lengths, character_map)
    mb_src, mb_tgt = mb_src.transpose_(0, 1), mb_tgt.transpose_(0, 1)

    return mb_src, mb_tgt


def pad_minibatch(minibatchwords, src_word_lengths, tgt_word_lengths, character_map):
    pad_token = character_map[PADDING]
    src_longest_word, tgt_longest_word = max(src_word_lengths), max(tgt_word_lengths)

    src_padded_mb = np.ones((len(minibatchwords), src_longest_word)) * pad_token
    tgt_padded_mb = np.ones((len(minibatchwords), tgt_longest_word)) * pad_token

    for i, x_len in enumerate(src_word_lengths):
        src_sequence = minibatchwords[i]['SOURCE_TENSOR']
        src_padded_mb[i, 0:x_len] = src_sequence[:x_len]

    for i, x_len in enumerate(tgt_word_lengths):
        tgt_sequence = minibatchwords[i]['TARGET_TENSOR']
        tgt_padded_mb[i, 0:x_len] = tgt_sequence[:x_len]

    src_padded_mb = torch.from_numpy(src_padded_mb).type(torch.LongTensor)
    tgt_padded_mb = torch.from_numpy(tgt_padded_mb).type(torch.LongTensor)

    return src_padded_mb, tgt_padded_mb


if __name__=='__main__':
    from paths import data_dir
