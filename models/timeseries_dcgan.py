import os

import torch
from torch.utils.data import Dataset
from model import DCGAN
import pandas as pd
import tensorflow as tf
from ops import *

import sys
sys.path.insert(0, '.')
from data import LGITDataset

tf.enable_eager_execution()

class TimeSeriesDCGAN(DCGAN):
  def __init__(self, *args, **kwargs):
    super(TimeSeriesDCGAN, self).__init__(*args, **kwargs)
  
  def load_time_data(self):
    data_dir = os.path.join(self.data_dir, self.dataset_name)
    data_df = pd.read_csv(os.path.join(data_dir, 'time.csv'))
    columns_df = pd.read_csv(os.path.join(data_dir, 'columns.csv'))
    dataset = LGITDataset(data_df, columns_df, batch_first = True)
    
    self.x_dim = dataset.get_dims()
    print('shape: ', dataset.input_data.shape)
    return dataset.input_data

  def discriminator(self, x, y=None):
    # we dont use y label for this version
    if not self.y_dim:
      x = tf.reshape(x, [self.batch_size, -1])
      dim_lat = 64
      dim_hidden = 32
      mlp_num_layers = 3

      start = 0
      end = start + self.x_dim['tab']
      x_tab = x[:, start:end]

      x_tab_repr = mlp(
        x_tab,
        d_in = end - start,
        d_out = dim_lat,
        d_layers = [dim_hidden] * mlp_num_layers,
        name='d_mlp',
        dropout=False
      )

      start = end
      end = start + self.x_dim['sa'] * 2
      
      x_sa = tf.reshape(
          x[:, start:end],
          [self.batch_size, self.x_dim['sa'], 2]
        )
      x_sa_repr = lstm(
        x_sa,
        d_in = end - start,
        d_hidden = dim_hidden,
        d_out = dim_lat,
        name='d_sa_lstm'
      )

      start = end
      end = start + self.x_dim['sb'] * 6
      x_sb = tf.reshape(
          x[:, start:end],
          [self.batch_size, self.x_dim['sb'], 6]
        )
      x_sb_repr = lstm(
        x_sb,
        d_in = end - start,
        d_hidden = dim_hidden,
        d_out = dim_lat,
        name='d_sb_lstm'
      )
      x_repr = concat([x_tab_repr, x_sa_repr, x_sb_repr], 1)
      out = linear(
        x_repr,
        1,
        scope='d_head'
      )
      return tf.nn.sigmoid(out), out
    else:
      raise NotImplementedError("we dont use y label for this version")

  def generator(self, z, y=None):
    # we dont use y label for this version
    with tf.variable_scope("generator") as scope:
      s_h, s_w = self.output_height, self.output_width
      s_h2 = s_h4 = s_h # 1d conv used
      s_w2, s_w4 = int(s_w / 2), int(s_w / 4)
      h0 = tf.nn.relu(self.g_bn0(linear(z, self.gfc_dim, 'g_h0_lin')))
      h1 = tf.nn.relu(self.g_bn1(linear(h0, self.gf_dim * 2 * s_h4 * s_w4, 'g_h1_lin')))
      h1 = tf.reshape(h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2])
      h2 = tf.nn.relu(self.g_bn2(deconv2d(h1, [self.batch_size, s_h2, s_w2, self.gf_dim * 2], name='g_h2')))

      return tf.nn.sigmoid(deconv2d(h2, [self.batch_size, s_h, s_w, self.c_dim], name='g_h3'))

