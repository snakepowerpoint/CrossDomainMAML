import numpy as np
import torch
import torch.optim
import os

from methods import backbone
from methods.backbone import model_dict
from data.datamgr import SimpleDataManager, SetDataManager
from methods.mamlBaseline import MAMLBaseline
from options import parse_args, get_resume_file, load_warmup_state


def train(base_loader, val_loader, model, start_epoch, stop_epoch, params):

  # for validation
  max_acc = 0
  total_it = 0

  # start
  for epoch in range(start_epoch, stop_epoch):
    model.train()
    total_it = model.train_loop(epoch, base_loader, total_it) 
    
    model.eval()
    acc = model.test_loop(val_loader, total_it)
    if acc > max_acc :
      print("best model! save...")
      max_acc = acc
      outfile = os.path.join(params.checkpoint_dir, 'best_model.tar')
      torch.save({'epoch':epoch, 'state':model.state_dict()}, outfile)
    else:
      print("GG! best accuracy {:f}".format(max_acc))

    if ((epoch + 1) % params.save_freq==0) or (epoch==stop_epoch-1):
      outfile = os.path.join(params.checkpoint_dir, '{:d}.tar'.format(epoch))
      torch.save({'epoch':epoch, 'state':model.state_dict()}, outfile)

  return model


# --- main function ---
if __name__=='__main__':

  # set numpy random seed
  np.random.seed(10)

  # parser argument
  params = parse_args('train')
  print('--- baseline training: {} ---\n'.format(params.name))
  print(params)

  # output and tensorboard dir
  params.tf_dir = '%s/log/%s'%(params.save_dir, params.name)
  params.checkpoint_dir = '%s/checkpoints/%s'%(params.save_dir, params.name)
  if not os.path.isdir(params.checkpoint_dir):
    os.makedirs(params.checkpoint_dir)

  # dataloader
  print('\n--- prepare dataloader ---')
  if params.dataset == 'multi':
    print('  train with multiple seen domains (unseen domain: {})'.format(params.testset))
    datasets = ['miniImagenet', 'cars', 'places', 'cub', 'plantae']
    datasets.remove(params.testset)
    base_file = [os.path.join(params.data_dir, dataset, 'base.json') for dataset in datasets]
    val_file  = os.path.join(params.data_dir, 'miniImagenet', 'val.json')
  else:
    print('  train with single seen domain {}'.format(params.dataset))
    base_file  = os.path.join(params.data_dir, params.dataset, 'base.json')
    val_file   = os.path.join(params.data_dir, params.dataset, 'val.json')

  # model
  print('\n--- build model ---')
  if 'Conv' in params.model:
    image_size = 84
  else:
    image_size = 224

  if params.method in ['maml_baseline'] :
    print('  training the {} with backbone {}'.format(params.method, params.model))
    n_query = 15
    train_few_shot_params = dict(n_way=params.train_n_way, n_support=params.n_shot)
    base_datamgr          = SetDataManager(image_size, n_query=n_query, n_eposide=100,  **train_few_shot_params)
    base_loader           = base_datamgr.get_data_loader(base_file, aug=params.train_aug)

    test_few_shot_params  = dict(n_way=params.test_n_way, n_support=params.n_shot)
    val_datamgr           = SetDataManager(image_size, n_query=n_query, n_eposide=100, **test_few_shot_params)
    val_loader            = val_datamgr.get_data_loader(val_file, aug=False)
  
    # base_datamgr    = SimpleDataManager(image_size, batch_size=16)
    # base_loader     = base_datamgr.get_data_loader(base_file , aug=params.train_aug )
    # val_datamgr     = SimpleDataManager(image_size, batch_size=64)
    # val_loader      = val_datamgr.get_data_loader(val_file, aug=False)
  else:
    raise ValueError('Unknown method')
  model = MAMLBaseline(params, tf_path=params.tf_dir)
  model.cuda()

  # load model
  start_epoch = params.start_epoch
  stop_epoch = params.stop_epoch
  if params.resume != '':
    resume_file = get_resume_file('%s/checkpoints/%s'%(params.save_dir, params.resume), params.resume_epoch)
    if resume_file is not None:
      tmp = torch.load(resume_file)
      start_epoch = tmp['epoch']+1
      model.load_state_dict(tmp['state'])
      print('  resume the training with at {} epoch (model file {})'.format(start_epoch, params.resume))
  elif True:
    pass
  elif params.method == 'maml_baseline':
    if params.warmup == 'gg3b0':
      raise Exception('Must provide the pre-trained feature encoder file using --warmup option!')
    state = load_warmup_state('%s/checkpoints/%s'%(params.save_dir, params.warmup), params.method)
    model.model.feature.load_state_dict(state, strict=False)

  # training
  print('\n--- start the training ---')
  model = train(base_loader, val_loader, model, start_epoch, stop_epoch, params)
