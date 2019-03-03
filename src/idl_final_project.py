from random import choice, random, shuffle
import numpy as np
import torch, sys
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from data import read_datasets, WORD_BOUNDARY, UNK
from paths import data_dir


#--- hyperparameters ---
N_EPOCHS = 50
BATCH_SIZE = 5




def get_minibatch(minibatchwords, character_map, languages):
    src_word_lengths = [len(x['SOURCE_TENSOR']) for x in minibatchwords]
    tgt_word_lengths = [len(x['TARGET_TENSOR']) for x in minibatchwords]

    mb_src, mb_tgt = pad_minibatch(minibatchwords, src_word_lengths, tgt_word_lengths)
    mb_src, mb_tgt = mb_src.transpose_(0, 1), mb_tgt.transpose_(0, 1)

    return mb_src, mb_tgt


def pad_minibatch(minibatchwords, src_word_lengths, tgt_word_lengths):
    pad_token = character_map[WORD_BOUNDARY]
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

    languages = {'-fin':'finnish', '-ger':'german', '-nav':'navajo'}

    if len(sys.argv) < 2:
        print('Usage:', sys.argv[0], '[ -fin | -ger | -nav ]')
        sys.exit(2)
    elif len(sys.argv) == 2 and sys.argv[1] in languages.keys():
        lang_choice = languages[sys.argv[1]]
        print('Training a model for ' + lang_choice.capitalize())
    else:
        print('Usage:', sys.argv[0], '[ -fin | -ger | -nav ]')
        sys.exit(2)


    data, character_map = read_datasets(lang_choice + '-task1',data_dir)
    trainset = [datapoint for datapoint in data['training']]

    for epoch in range(N_EPOCHS):

        shuffle(trainset)
        # trainset = sorted(trainset, key=lambda x: len(x['SOURCE_TENSOR'])) # Sort by length
               
        for i in range(0,len(trainset),BATCH_SIZE):
            minibatchwords = trainset[i:i+BATCH_SIZE]
            mb_src, mb_tgt = get_minibatch(minibatchwords, character_map, languages)



