import argparse
import os
import numpy as np
import torch

from torch.autograd import Variable
from torch import nn, optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from dataset import AgDataset, preprocess

GPU_NUM = True

def collate_fn(data: list):
  review = []
  label = []
  for datum in data:
      review.append(datum[0])
      label.append(datum[1])
  return review, np.array(label)

def position_encoding_init(n_position, d_pos_vec):
  ''' Init the sinusoid position encoding table '''

  # keep dim 0 for padding token position encoding zero vector
  position_enc = np.array([
      [pos / np.power(10000, 2 * (j // 2) / d_pos_vec) for j in range(d_pos_vec)]
      if pos != 0 else np.zeros(d_pos_vec) for pos in range(n_position)])

  position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2]) # dim 2i
  position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2]) # dim 2i+1
  return torch.from_numpy(position_enc).type(torch.FloatTensor)

from tcn import TemporalConvNet
class TCN(nn.Module):
  def __init__(self, embedding_dim: int, max_length: int, channel=200, level=3,
               kernel_size=3, dropout=0.2, emb_dropout=0., tied_weights=False, attention=False):
    super(TCN, self).__init__()

    self.channel = channel
    self.channels = [channel] * level

    self.embedding_dim = embedding_dim
    self.character_size = 252
    self.max_length = max_length

    self.embeddings = nn.Embedding(self.character_size, self.embedding_dim, padding_idx=0)
    self.pe = nn.Embedding(self.max_length, self.embedding_dim, padding_idx=0)
    self.pe.weight.data.copy_(position_encoding_init(self.max_length, self.embedding_dim))
    self.pe.weight.requires_grad = False
    self.tcn = TemporalConvNet(embedding_dim, self.channels, kernel_size, dropout=dropout, max_length=max_length, attention=attention)

  def forward(self, inputs, lens):
    data_in_torch = Variable(torch.from_numpy(np.array(inputs)).long())
    len_in_torch = Variable(torch.from_numpy(np.array(lens)).long())
    if GPU_NUM:
      data_in_torch = data_in_torch.cuda()
      len_in_torch = len_in_torch.cuda()
    embeds = self.embeddings(data_in_torch)
    pe = self.pe(len_in_torch)
    embeds += pe
    #output = self.tcn(embeds)
    #return output
    output = self.tcn(embeds.transpose(1,2)).transpose(1,2)
    return output.contiguous()

class TNT(nn.Module):
  def __init__(self, embedding_dim: int, max_length: int,
               channel_size=200, T_size=16, level=3, attention=False):
    super(TNT, self).__init__()
    self.tcn = TCN(embedding_dim, max_length, channel=channel_size, level=level, attention=attention)
    self.embedding_dim = embedding_dim
    self.max_length = max_length
    self.output_dim = 1

    # model T
    self.fc1 = nn.Linear(self.max_length * channel_size, T_size)
    self.act1 = nn.ReLU()
    self.fc2 = nn.Linear(T_size, 5)

    self.init_weights()

  def init_weights(self):
    self.fc1.bias.data.fill_(0)
    nn.init.xavier_uniform(self.fc1.weight, gain=np.sqrt(2))
    self.fc2.bias.data.fill_(0)
    nn.init.xavier_uniform(self.fc2.weight, gain=np.sqrt(2))

  def forward(self, inputs, lens):
    sent = self.tcn(inputs, lens)
    sent = sent.view(sent.size(0), -1)
    net = self.act1(self.fc1(sent))
    out = self.fc2(net)
    return out

args = argparse.ArgumentParser()
args.add_argument('--mode', type=str, default='train')
# User options
args.add_argument('--epochs', type=int, default=30)
args.add_argument('--batch', type=int, default=20)
args.add_argument('--strmaxlen', type=int, default=1000)
args.add_argument('--embedding', type=int, default=8)
args.add_argument('--lr', type=float, default=1e-4)
args.add_argument('--convchannel', type=int, default=200)
args.add_argument('--tsize', type=int, default=1000)
args.add_argument('--lrstep', type=int, default=1000)
args.add_argument('--level', type=int, default=3)
args.add_argument('--attention', type=bool, default=False)
config = args.parse_args()

DATASET_PATH = './data/'
dataset = AgDataset(DATASET_PATH, config.strmaxlen, mode=config.mode)
model = TNT(config.embedding, config.strmaxlen, channel_size=config.convchannel, T_size=config.tsize, level=config.level, attention=config.attention)
if GPU_NUM:
  model = model.cuda()

if config.mode == 'train':
  train_loader = DataLoader(dataset=dataset, batch_size=config.batch, shuffle=True, collate_fn=collate_fn, num_workers=2)
elif config.mode == 'test':
  model.load_state_dict(torch.load('model.pkl'))
  test_loader = DataLoader(dataset=dataset, batch_size=config.batch, shuffle=True, collate_fn=collate_fn, num_workers=2)

if config.mode == 'train':
  parms = filter(lambda p: p.requires_grad, model.parameters())

  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(parms, lr=config.lr)
  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=config.lrstep, factor=0.95, verbose=False, mode='min')

  total_batch = len(train_loader)
  # epoch마다 학습을 수행합니다.
  for epoch in range(config.epochs):
    avg_loss = 0.0
    for i, (data, labels) in enumerate(train_loader):
      data1, lens1 = zip(*data)
      predictions = model(data1, lens1)
      label_vars = Variable(torch.from_numpy(labels).long())
      if GPU_NUM:
        label_vars = label_vars.cuda()
      loss = criterion(predictions, label_vars)
      if GPU_NUM:
        loss = loss.cuda()

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      scheduler.step(loss.data[0])
      avg_loss += loss.data[0]
      if i == 0 or i % (total_batch/10) == 0:
        print('Batch : ', i + 1, '/', total_batch, ', Loss in this minibatch: ', loss.data[0])
    print('epoch:', epoch, ' train_loss:', float(avg_loss/total_batch))
  torch.save(model.state_dict(), 'model.pkl')

elif config.mode == 'test':
  model.eval()

  wrong = 0
  correct = 0

  for i, (data, labels) in enumerate(test_loader):
    data1, lens1 = zip(*data)
    preds = model(data1, lens1).data.cpu()
    for j in range(len(preds)):
      if np.argmax(preds[j]) == labels[j]:
        correct += 1
      else:
        wrong += 1

  print("test accuracy: ", float(correct)/(wrong + correct))
