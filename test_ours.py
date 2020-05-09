import numpy as np
import os
import random
import torch
from data.datamgr import SetDataManager, SimpleDataManager
from options import parse_args, get_resume_file, load_warmup_state
from options import get_best_file, get_assigned_file
from methods.maml import MAML
from methods import protonet
from methods import matchingnet
from methods import relationnet
from methods import gnnnet
from methods.backbone import model_dict
from methods import backbone
from methods import gnn


def test(task, model, n_iter, n_sub_query, params):
  # weight
  # model_params, _ = split_model_parameters(model)

  # train loop: update model using support set
  n_support = params.n_shot
  support = task[:, :n_support, :, :, :]
  for _ in range(n_iter):
    # shuffle support
    support = support[:, torch.randperm(support.size(1))]

    # model setting and forward
    model.train()
    model.model.n_query = n_sub_query
    model.model.n_support = n_support - n_sub_query
    _, model_loss = model.model.set_forward_loss(support)

    # # update model parameters according to model_loss
    # meta_grad = torch.autograd.grad(model_loss, model_params, create_graph=True)
    # for k, weight in enumerate(model_params):
    #   if weight.fast is None:
    #     weight.fast = weight - model.model_optim.param_groups[0]['lr']*meta_grad[k]
    #   else:
    #     weight.fast = weight.fast - model.model_optim.param_groups[0]['lr']*meta_grad[k]
    # meta_grad = [g.detach() for g in meta_grad]
    
    # optimize model
    model.model_optim.zero_grad()
    model_loss.backward()
    model.model_optim.step()

  # validate
  model.eval()
  model.model.n_support = n_support
  n_query = task.size(1) - n_support
  model.model.n_query = n_query
  
  scores, _ = model.model.set_forward_loss(task)
  pred = scores.data.cpu().numpy().argmax(axis = 1)
  y = np.repeat(range( params.test_n_way ), n_query )
  acc = np.mean(pred == y)*100
  
  return acc

# split the parameters of feature-wise transforamtion layers and others
def split_model_parameters(model):
  model_params = []
  ft_params = []
  for n, p in model.named_parameters():
    n = n.split('.')
    if n[-1] == 'gamma' or n[-1] == 'beta':
      ft_params.append(p)
    else:
      model_params.append(p)
  return model_params, ft_params


# --- main function ---
if __name__=='__main__':

  # # set numpy random seed
  # np.random.seed(10)

  # parse argument
  params = parse_args('test')
  print('Testing! {} shots on {} dataset with {} epochs of {}({})'.format(params.n_shot, params.dataset, params.save_epoch, params.name, params.method))
  print(params)

  print('\nStage 1: prepare test dataset and model')
  # dataloader 
  print('\n--- build dataset ---')
  if 'Conv' in params.model:
    image_size = 84
  else:
    image_size = 224
  n_task   = params.n_task  # 1000
  n_query  = 15
  split    = params.split  # novel
  loadfile = os.path.join(params.data_dir, params.dataset, split + '.json')
  test_few_shot_params = dict(n_way = params.test_n_way, n_support = params.n_shot)
  datamgr              = SetDataManager(image_size, n_query = n_query, n_eposide = n_task, **test_few_shot_params)
  data_loader          = datamgr.get_data_loader(loadfile, aug = False)  

  # model
  print('\n--- build MAML model ---')
  params.tf_dir = '%s/log/%s'%(params.save_dir, params.name)
  params.checkpoint_dir = '%s/checkpoints/%s'%(params.save_dir, params.name)
  if not os.path.isdir(params.checkpoint_dir):
    os.makedirs(params.checkpoint_dir)

  model = MAML(params, tf_path=params.tf_dir)
  model.cuda()

  print('\nStage 2: evaluate')
  # resume model
  if params.save_epoch != -1:
    modelfile   = get_assigned_file(params.checkpoint_dir, params.save_epoch)
  else:
    modelfile   = get_best_file(params.checkpoint_dir)
  # _ = model.resume(modelfile)

  # start_epoch = params.start_epoch
  # stop_epoch = params.stop_epoch
  # if params.resume != '':
  #   resume_file = get_resume_file('%s/checkpoints/%s'%(params.save_dir, params.resume), params.resume_epoch)
  #   if resume_file is not None:
  #     start_epoch = model.resume(resume_file)
  #     print('  resume the training with at {} epoch (model file {})'.format(start_epoch, params.resume))
  #   else:
  #     raise ValueError('No resume file')
  # # load pre-trained feature encoder
  # else:
  #   if params.warmup == 'gg3b0':
  #     raise Exception('Must provide pre-trained feature-encoder file using --warmup option!')
  #   model.model.feature.load_state_dict(load_warmup_state('%s/checkpoints/%s'%(params.save_dir, params.warmup), params.method), strict=False)

  # start evaluate
  print('\n--- start the testing ---')
  acc_all = []
  n_iter = params.n_iter
  data_generator = iter(data_loader)

  for i in range(n_task):
    task = next(data_generator)[0]
    n_sub_query = 1
    _ = model.resume(modelfile)
    acc = test(task, model, n_iter, n_sub_query, params)
    acc_all.append(acc)

  # statics
  print('\n--- get statics ---')
  acc_all = np.asarray(acc_all)
  acc_mean = np.mean(acc_all)
  acc_std = np.std(acc_all)
  print('  %d test task, %d iteration: Acc = %4.2f%% +- %4.2f%%' % (n_task, n_iter, acc_mean, 1.96* acc_std/np.sqrt(n_task)))
