#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 niezhaochang <niezhaochang@amax3>
#
# Distributed under terms of the MIT license.

"""
build the embedding of static matrix [Batch*Field_size]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class StEmb(nn.Module):
    """
    batch_size: batch_size 
    field_size: length of feature_sizes, the number of feature field
    feature_sizes: an array of feature size, e.g. age=[0,1,2,3,4], feature_size_{age}=5
    embedding_size: size of embedding
    dropout: prob for dropout, set None if no dropout
    use_cuda: bool, True for gpu or False for cpu
    """
    def __init__(self, batch_size, field_size, total_feature_size, embedding_size=4, dropout_rate=None, use_cuda=True):
        super(StEmb, self).__init__()
        self.batch_size = batch_size
        self.field_size = field_size
        self.total_feature_size = total_feature_size
        self.embedding_size = embedding_size
        self.dropout_rate = dropout_rate
        self.use_cuda = use_cuda

        # initial layer
        #  self.embeddings = nn.ModuleList([nn.Embedding(feature_size, self.embedding_size) for feature_size in self.feature_sizes])
        self.embeddings = nn.Embedding(self.total_feature_size, self.embedding_size, padding_idx=0)

        self.is_dropout = False
        if self.dropout_rate is not None:
            self.is_dropout = True
            self.dropout = nn.Dropout(p=self.dropout_rate)

    def forward(self, static_ids):
        """
        input: relative id 
        static_ids: Batch_size * Field_size
        return: Batch_size * Field_size * Embedding_size
        """

        # B*F
        static_ids_tensor = torch.autograd.Variable(static_ids)

        # embedding layer [B*F]*E
        static_embeddings_tensor = self.embeddings(static_ids_tensor)

        # dropout
        if self.is_dropout:
            static_embeddings_tensor = self.dropout(static_embeddings_tensor)

        return static_embeddings_tensor


if __name__ == '__main__':
    # test
    batch_size = 2
    field_size = 2
    total_feature_size = 6
    ids = [[1,5],[2,5]]


    stEmb = StEmb(batch_size, field_size, total_feature_size)
    stEmb = stEmb.cuda()

    st_embeddings = stEmb(ids)

    print st_embeddings



