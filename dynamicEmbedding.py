#! /usr/bin/env python


"""
build the embedding of dynamic matrix [Batch*Field_size*Dynamic_Feature_Size]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable



class DyEmb(nn.Module):
    def __init__(self, field_size, max_feature_size, total_feature_size, embedding_size=4, method='avg'):
        """
        field_size: length of feature_sizes, the number of feature field
        total_feature_size: total feature size
        feature_sizes: an array of feature size, e.g. age=[0,1,2,3,4], feature_size_{age}=5
        embedding_size: size of embedding
        method: 'avg' or 'sum'
        use_cuda: bool, True for gpu or False for cpu
        """
        super(DyEmb, self).__init__()

        assert method in ['avg', 'sum']

        self.field_size = field_size
        self.total_feature_size = total_feature_size
        self.max_feature_size = max_feature_size
        self.embedding_size = embedding_size
        self.method = method

        self.embeddings = nn.Embedding(self.total_feature_size, self.embedding_size, padding_idx=0)


    def forward(self, dynamic_ids, dynamic_lengths):
        """
        input: relative id 
        dynamic_ids: Batch_size * [Field_size * Max_feature_size], Variable LongTensor
        dynamic_lengths: Batch_size * Field_size, Variable LongTensor
        return: Batch_size * Field_size * Embedding_size
        """


        assert self.max_feature_size * self.field_size == dynamic_ids.size()[-1]



        # B*[F*M] -> [B*F]*M
        dynamic_ids_tensor = dynamic_ids.view(-1, self.max_feature_size)
        # B*F -> [B*F]
        dynamic_lengths_tensor = dynamic_lengths.view(-1)

        # embedding layer [B*F]*M*E
        dynamic_embeddings_tensor = self.embeddings(dynamic_ids_tensor)



        # average [B*F]*M*E --AVG--> [B*F]*E
        dynamic_sum_embedding = torch.sum(dynamic_embeddings_tensor, 1)

        # [B*F]*E -> B*F*E
        if self.method == 'avg':
            dynamic_lengths_tensor = torch.autograd.Variable(dynamic_lengths_tensor.data.float())
            dynamic_lengths_tensor = dynamic_lengths_tensor.view(-1, 1).expand_as(dynamic_sum_embedding)
            dynamic_avg_embedding = dynamic_sum_embedding / dynamic_lengths_tensor
            dynamic_embeddings = dynamic_avg_embedding.view(-1, self.field_size, self.embedding_size)
        else:
            dynamic_embeddings = dynamic_sum_embedding.view(-1, self.field_size, self.embedding_size)

        return dynamic_embeddings

#
# if __name__ == '__main__':
#     # test
#     batch_size = 2
#     field_size = 2
#     feature_sizes = 10
#     max_feature_size = 5
#     ids = [[[2,1,3,0,0,5,0,0,0,0]],[[2,2,0,0,0,5,5,5,5,5]]]
#     lengths = [[3,1],[2,5]]
#
#     ids = Variable(torch.LongTensor(ids))
#     lengths = Variable(torch.LongTensor(lengths))
#
#     #  dyEmb = DyEmb(batch_size, field_size, feature_sizes, embedding_size=1, method='sum')
#     dyEmb = DyEmb(field_size, max_feature_size, feature_sizes)
#
#     avg_embeddings = dyEmb(ids, lengths)
#
#     print avg_embeddings


