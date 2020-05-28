from methods import backbone
from methods import mamlnet
from methods.backbone import model_dict

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
import numpy as np


# --- conventional supervised training ---
class MAMLBaseline(nn.Module):
  def __init__(self, params, tf_path=None):
  # def __init__(self, model_func, n_way, n_support, params, tf_path=None):
    super(MAMLBaseline, self).__init__()

    self.tf_writer = SummaryWriter(log_dir=tf_path) if tf_path is not None else None

    self.n_task = 4
    self.task_update_num = 5
    self.train_lr = 0.05

    # get maml model and enable L2L(maml) training
    train_few_shot_params = dict(n_way=params.train_n_way, n_support=params.n_shot)
    backbone.ConvBlock.maml = True
    backbone.SimpleBlock.maml = True
    backbone.ResNet.maml = True
    if params.method in ['maml_baseline']:
      mamlnet.ClassifierModule.maml = True
      if params.model == 'Conv4':
        feature_model = backbone.Conv4NP
      elif params.model == 'Conv6':
        feature_model = backbone.Conv6NP
      else:
        feature_model = model_dict[params.model]
      model = mamlnet.MAMLNet(feature_model, tf_path=params.tf_dir, **train_few_shot_params)
    self.model = model
    print('  train with {} framework'.format(params.method))
    
    # optimizer
    model_params, ft_params = self.split_model_parameters()
    self.model_optim = torch.optim.Adam(model_params, lr=params.lr)

    # total epochs
    self.total_epoch = params.stop_epoch

  # split the parameters of feature-wise transforamtion layers and others
  def split_model_parameters(self):
    model_params = []
    ft_params = []
    for n, p in self.model.named_parameters():
      n = n.split('.')
      if n[-1] == 'gamma' or n[-1] == 'beta':
        ft_params.append(p)
      else:
        model_params.append(p)
    return model_params, ft_params

  def train_loop(self, epoch, train_loader, total_it):
    print_freq = len(train_loader) / 10
    avg_loss = 0.
    task_count = 0
    loss_all = []
    self.model_optim.zero_grad()

    for i, (x, _) in enumerate(train_loader):

      # clear fast weight
      fast_parameters = self.split_model_parameters()[0]
      for weight in self.split_model_parameters()[0]:
        weight.fast = None
      self.zero_grad()

      # classification loss on support
      # self.model.train()
      self.model.n_query = x.size(1) - self.model.n_support
      support = x[:, :self.model.n_support, :, :, :]
      y = torch.from_numpy(np.repeat(range(self.model.n_way), self.model.n_support))
      
      for _ in range(self.task_update_num):
        _, fast_loss = self.model.set_forward_loss(support, y)

        # update model parameters according to fast loss
        # meta_grad = torch.autograd.grad(fast_loss, fast_parameters)
        meta_grad = torch.autograd.grad(fast_loss, fast_parameters, create_graph=True)
        meta_grad = [g.detach() for g in meta_grad]  # do not calculate gradient of gradient if using first order approximation
        fast_parameters = []
        for k, weight in enumerate(self.split_model_parameters()[0]):
          if weight.fast is None:
            weight.fast = weight - self.train_lr * meta_grad[k]
          else:
            weight.fast = weight.fast - self.train_lr * meta_grad[k]
          fast_parameters.append(weight.fast)      

      # classification loss with updated model
      # self.model.eval()
      query = x[:, self.model.n_support:, :, :, :]
      y = torch.from_numpy(np.repeat(range(self.model.n_way), self.model.n_query))
      _, model_loss = self.model.set_forward_loss(query, y)

      avg_loss = avg_loss + model_loss.item() # data[0]
      loss_all.append(model_loss)

      task_count += 1
      
      # optimize model: MAML update several tasks at one time
      if task_count == self.n_task:
        loss_q = torch.stack(loss_all).sum(0)
        # self.model_optim.zero_grad()
        loss_q.backward()
        self.model_optim.step()
        task_count = 0
        loss_all = []
      self.model_optim.zero_grad()
      
      if (i + 1) % print_freq==0:
        print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i + 1, len(train_loader), avg_loss/float(i+1)))
      if (total_it + 1) % 10 == 0:
        self.tf_writer.add_scalar('loss', model_loss.item(), total_it + 1)
      total_it += 1
    return total_it

  def test_loop(self, test_loader, total_it, record=None):
    loss = 0.
    acc_all = []

    iter_num = len(test_loader)
    for i, (x, _) in enumerate(test_loader):

      # clear fast weight
      for weight in self.split_model_parameters()[0]:
        weight.fast = None

      # classification loss on support
      # self.model.train()
      self.model.n_query = x.size(1) - self.model.n_support
      support = x[:, :self.model.n_support, :, :, :]
      y = torch.from_numpy(np.repeat(range(self.model.n_way), self.model.n_support))
      _, fast_loss = self.model.set_forward_loss(support, y)

      # update model parameters according to model_loss
      meta_grad = torch.autograd.grad(fast_loss, self.split_model_parameters()[0], create_graph=True)
      meta_grad = [g.detach() for g in meta_grad]
      for k, weight in enumerate(self.split_model_parameters()[0]):
        weight.fast = weight - self.train_lr * meta_grad[k]

      # classification loss with updated model
      with torch.no_grad():
        # self.model.eval()
        query = x[:, self.model.n_support:, :, :, :]
        y = torch.from_numpy(np.repeat(range(self.model.n_way), self.model.n_query))
        scores, model_loss = self.model.set_forward_loss(query, y)
      
      pred = scores.data.cpu().numpy().argmax(axis=1)
      y = np.repeat(range(self.model.n_way), self.model.n_query)

      acc_all.append(np.mean(pred == y)*100)
      loss += model_loss.item()

    acc_all = np.asarray(acc_all)
    acc_mean = np.mean(acc_all)
    acc_std = np.std(acc_all)
    loss_mean = loss/iter_num
    print('--- %d Loss = %.6f ---' % (iter_num,  loss_mean))
    print('--- %d Test Acc = %4.2f%% +- %4.2f%% ---' % (iter_num,  acc_mean, 1.96 * acc_std/np.sqrt(iter_num)))

    self.tf_writer.add_scalar('MAML/val_acc', acc_mean, total_it + 1)
    self.tf_writer.add_scalar('MAML/val_loss', loss/iter_num, total_it + 1)

    if self.tf_writer is not None:
      self.tf_writer.add_scalar('MAML/val/loss', loss_mean, total_it)
      self.tf_writer.add_scalar('MAML/val/acc', acc_mean, total_it)
    return acc_mean

  # def test_loop(self, val_loader):
  #   return -1 #no validation, just save model during iteration

  def cuda(self):
    self.model.cuda()
