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
from tensorboardX import SummaryWriter
from tqdm import tqdm

def test(base_task, test_task, model, n_iter, n_sub_query, params):
  # model optimizer
  if params.opt == 'sgd':
    model_params, _ = model.split_model_parameters()
    model.model_optim = torch.optim.SGD(model_params, lr=params.lr)
  elif params.opt == 'adam':
    model_params, _ = model.split_model_parameters()
    model.model_optim = torch.optim.Adam(model_params, lr=params.lr)
  else:
    pass  

  # train loop: update model using support set
  n_support = params.n_shot
  base_support = base_task[:, :n_support, :, :, :]
  test_support = test_task[:, :n_support, :, :, :]
  for _ in range(n_iter):
    # shuffle support
    base_support = base_support[:, torch.randperm(base_support.size(1))]
    test_support = test_support[:, torch.randperm(test_support.size(1))]

    # model setting and forward
    model.train()
    model.model.n_query = n_sub_query
    model.model.n_support = n_support - n_sub_query
    _, base_loss = model.model.set_forward_loss(base_support)
    _, test_loss = model.model.set_forward_loss(test_support)
    total_loss = (1-params.beta) * base_loss + params.beta * test_loss

    # optimize model
    model.model_optim.zero_grad()
    total_loss.backward()
    model.model_optim.step()

  # validate
  model.eval()
  model.model.n_support = n_support
  n_query = base_task.size(1) - n_support
  model.model.n_query = n_query
  
  y = np.repeat(range( params.test_n_way ), n_query )
  base_scores, _ = model.model.set_forward_loss(base_task)
  pred = base_scores.data.cpu().numpy().argmax(axis = 1)
  base_acc = np.mean(pred == y)*100

  test_scores, _ = model.model.set_forward_loss(test_task)
  pred = test_scores.data.cpu().numpy().argmax(axis = 1)
  test_acc = np.mean(pred == y)*100
  
  return base_acc, test_acc

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
  print('Testing! {} shots on {} and {} with {} epochs of {}({})'.format(params.n_shot, params.dataset, params.testset, params.save_epoch, params.name, params.method))
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
  base_loadfile = os.path.join(params.data_dir, params.dataset, split + '.json')
  test_loadfile = os.path.join(params.data_dir, params.testset, split + '.json')
  test_few_shot_params = dict(n_way = params.test_n_way, n_support = params.n_shot)
  datamgr              = SetDataManager(image_size, n_query = n_query, n_eposide = n_task, **test_few_shot_params)
  
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
  
  # start evaluate
  print('\n--- start the testing ---')
  n_exp = params.n_exp
  n_iter = params.n_iter
  tf_path = '%s/log_test/%s_iter_%s_%s'%(params.save_dir, params.name, params.n_iter, params.opt)
  tf_writer = SummaryWriter(log_dir=tf_path) 
    
  # statics
  print('\n--- get statics ---')  
  for i in range(n_exp):
    acc_all = np.empty((n_task, 2))
    base_data_loader = datamgr.get_data_loader(base_loadfile, aug = False)
    test_data_loader = datamgr.get_data_loader(test_loadfile, aug = False)

    base_data_generator = iter(base_data_loader)
    test_data_generator = iter(test_data_loader)
    
    task_pbar = tqdm(range(n_task))
    for j in task_pbar:
      base_task = next(base_data_generator)[0]
      test_task = next(test_data_generator)[0]
      n_sub_query = 1
      _ = model.resume(modelfile)
      base_acc, test_acc = test(base_task, test_task, model, n_iter, n_sub_query, params)
      acc_all[j] = [base_acc, test_acc]

    acc_mean = np.mean(acc_all, axis=0)
    acc_std = np.std(acc_all, axis=0)

    base_acc, test_acc = acc_mean
    base_ci, test_ci = 1.96* acc_std/np.sqrt(n_task)

    print('  %d Exp: %d task, %d iter: Acc base = %4.2f%% +- %4.2f%%, test = %4.2f%% +- %4.2f%%' % (i+1, n_task, n_iter, base_acc, base_ci, test_acc, test_ci))
    
    tf_writer.add_scalar('base acc', base_acc, i + 1)
    tf_writer.add_scalar('base CI', base_ci, i + 1)
    tf_writer.add_scalar('test acc', test_acc, i + 1)
    tf_writer.add_scalar('test CI', test_ci, i + 1)

