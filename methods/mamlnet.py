# This code is modified from https://github.com/floodsung/LearningToCompare_FSL

from methods import backbone
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from methods.meta_template import MetaTemplate
import utils

class MAMLNet(MetaTemplate):
  def __init__(self, model_func, n_way, n_support, tf_path=None):
    super(MAMLNet, self).__init__(model_func, n_way, n_support, flatten=True, tf_path=tf_path)

    # loss function
    self.loss_fn = nn.CrossEntropyLoss()

    # # feature encoder
    # self.feature = model_func()  # ResNet10 (RestNet(nn.Module))

    # classifier
    self.classifier_module = ClassifierModule(self.feature.final_feat_dim, n_way)
    self.method = 'MAMLnet'

  def set_forward(self, x, is_feature=False):
    x = x.cuda()
    x = x.contiguous().view( self.n_way * x.size()[1], *x.size()[2:])
    out = self.feature.forward(x)
    scores = self.classifier_module.forward(out)
    return scores

  def set_forward_loss(self, x, y):
    # out = self.feature.forward(x)
    # scores = self.classifier_module.forward(out)
    scores = self.set_forward(x)
    y = y.cuda()
    loss = self.loss_fn(scores, y)
    return scores, loss


# --- Classifier module adopted in MAML ---
class ClassifierModule(nn.Module):
  maml = False
  def __init__(self, input_size, output_size):
    super(ClassifierModule, self).__init__()
    
    if self.maml:
      self.classifier = backbone.Linear_fw(input_size, output_size)
      self.classifier.bias.data.fill_(0)
    else:
      self.classifier = nn.Linear(input_size, output_size)
      self.classifier.bias.data.fill_(0)

  def forward(self, x):
    out = self.classifier(x)
    return out
