import os
import random

import numpy as np
from torch.utils.data import Dataset

from kor_char_parser import decompose_str_as_one_hot
import re

def get_one_hot(targets, nb_classes):
  return np.eye(nb_classes, dtype=np.float32)[np.array(targets).reshape(-1)]

class AgDataset(Dataset):
  def __init__(self, dataset_path: str, max_length: int, mode: str):
    """
    initializer

    :param dataset_path: 데이터셋 root path
    :param max_length: 문자열의 최대 길이
    """
    # 데이터, 레이블 각각의 경로
    data_review = os.path.join(dataset_path, 'ag_news-splitted-original', mode + '-description.txt')
    data_label = os.path.join(dataset_path, 'ag_news-splitted-original', mode + '-classes.txt')

    # 영화리뷰 데이터를 읽고 preprocess까지 진행합니다
    with open(data_review, 'rt', encoding='utf-8') as f:
      self.reviews = preprocess(f.readlines(), max_length)
    # 영화리뷰 레이블을 읽고 preprocess까지 진행합니다.
    with open(data_label) as f:
      self.labels = [np.float32(x) for x in f.readlines()]

  def __len__(self):
    return len(self.reviews)

  def __getitem__(self, idx):
    return self.reviews[idx], self.labels[idx]

def preprocess(data: list, max_length: int):
  vectorized_data = [decompose_str_as_one_hot(datum, warning=False) for datum in data]
  print("longest length: ", len(max(vectorized_data, key=len)))
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
