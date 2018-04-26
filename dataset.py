# -*- coding: utf-8 -*-

"""
Copyright 2018 NAVER Corp.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial
portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import os
import random

import numpy as np
from torch.utils.data import Dataset

from kor_char_parser import decompose_str_as_one_hot
import re

def get_one_hot(targets, nb_classes):
  return np.eye(nb_classes, dtype=np.float32)[np.array(targets).reshape(-1)]

class AgDataset(Dataset):
    """
    영화리뷰 데이터를 읽어서, tuple (데이터, 레이블)의 형태로 리턴하는 파이썬 오브젝트 입니다.
    """
    def __init__(self, dataset_path: str, max_length: int):
        """
        initializer

        :param dataset_path: 데이터셋 root path
        :param max_length: 문자열의 최대 길이
        """
        # 데이터, 레이블 각각의 경로
        data_review = os.path.join(dataset_path, 'ag_news-splitted', 'description.txt')
        data_label = os.path.join(dataset_path, 'ag_news-splitted', 'classes.txt')

        # 영화리뷰 데이터를 읽고 preprocess까지 진행합니다
        with open(data_review, 'rt', encoding='utf-8') as f:
            self.reviews = preprocess(f.readlines(), max_length)
        # 영화리뷰 레이블을 읽고 preprocess까지 진행합니다.
        with open(data_label) as f:
            #self.labels = [np.int(x) for x in f.readlines()]
            self.labels = [np.float32(x) for x in f.readlines()]
            #self.labels = [get_one_hot(np.int(x), 11) for x in f.readlines()]

        #tmp2= []
        #for i in range(len(self.labels)):
        #    tmp2.append(get_one_hot(self.labels[i], 11))

        #tmp1 = []
        #tmp2 = []
        #lc = [0] * 11
        #check = [0] * len(self.reviews) 

        #for times in range(9):
        #  for i in range(len(self.reviews)):
        #    if check[i] == 1:
        #      continue

        #    label = int(self.labels[i])
        #    if (label == 10 and lc[label] < 50000) or (lc[label] - list(sorted(lc))[times]) < 2:
        #      check[i] = 1
        #      lc[label]+=1
        #      tmp1.append(self.reviews[i])
        #      tmp2.append(self.labels[i])

        #print(lc)

        #self.reviews = tmp1
        #self.labels = tmp2

        ##self.reviews = tmp1
        ##self.labels = tmp2

        #print(len(self.labels))

    def __len__(self):
        """

        :return: 전체 데이터의 수를 리턴합니다
        """
        return len(self.reviews)

    def __getitem__(self, idx):
        """

        :param idx: 필요한 데이터의 인덱스
        :return: 인덱스에 맞는 데이터, 레이블 pair를 리턴합니다
        """
        return self.reviews[idx], self.labels[idx]

def ppp(string):
  string = re.sub(r'mv[0-9]{8}', ' V ', string)
  string = re.sub(r'ac[0-9]{8}', ' M ', string)
  return string

def preprocess(data: list, max_length: int):
    """
     입력을 받아서 딥러닝 모델이 학습 가능한 포맷으로 변경하는 함수입니다.
     기본 제공 알고리즘은 char2vec이며, 기본 모델이 MLP이기 때문에, 입력 값의 크기를 모두 고정한 벡터를 리턴합니다.
     문자열의 길이가 고정값보다 길면 긴 부분을 제거하고, 짧으면 0으로 채웁니다.

    :param data: 문자열 리스트 ([문자열1, 문자열2, ...])
    :param max_length: 문자열의 최대 길이
    :return: 벡터 리스트 ([[0, 1, 5, 6], [5, 4, 10, 200], ...]) max_length가 4일 때
    """
    vectorized_data = [decompose_str_as_one_hot(ppp(datum), warning=False) for datum in data]
    print("longest: ", len(max(vectorized_data, key=len)))
    zero_padding = np.zeros((len(data), max_length), dtype=np.int32)
    lens1 = np.zeros((len(vectorized_data), max_length), dtype=np.int32)
    for idx, seq in enumerate(vectorized_data):
        length = min(len(seq), max_length)
        lens1[idx] = np.pad(np.arange(length)+1, (0, max_length - length), 'constant')
        if length >= max_length:
            length = max_length
            zero_padding[idx, :length] = np.array(seq)[:length]
        else:
            zero_padding[idx, :length] = np.array(seq)
    return list(zip(zero_padding, lens1))
