import numpy as np
import os
import random
import torch
from data.datamgr import SetDataManager
from options import parse_args, get_resume_file, load_warmup_state
from methods.maml import MAML


# training iterations
def train(base_datamgr, base_set, val_loader, val_loader_nd, model, start_epoch, stop_epoch, params):
  
  # use miniImagenet as base (seen) domain and other dataset as meta (unseen) domain
  ps_set = 'miniImagenet'
  base_set.remove(ps_set)

  # for validation
  max_acc = 0
  total_it = start_epoch * 100

  # training
  for epoch in range(start_epoch,stop_epoch):

    # randomly split seen domains to pseudo-seen and pseudo-unseen domains
    if params.mix:
      pu_set = base_set
    else:
      pu_set = random.sample(base_set, k=1)
      pu_set_nd = random.sample(base_set, k=1)
      
    ps_loader = base_datamgr.get_data_loader(os.path.join(params.data_dir, ps_set, 'base.json'), aug=params.train_aug)
    ps_loader_nd = base_datamgr.get_data_loader(os.path.join(params.data_dir, ps_set, 'base.json'), aug=params.train_aug)
    pu_loader = base_datamgr.get_data_loader([os.path.join(params.data_dir, dataset, 'base.json') for dataset in pu_set], aug=params.train_aug)
    pu_loader_nd = base_datamgr.get_data_loader([os.path.join(params.data_dir, dataset, 'base.json') for dataset in pu_set_nd], aug=params.train_aug)
    
    # train loop
    model.train()
    total_it = model.trainall_loop(epoch, ps_loader, ps_loader_nd, pu_loader, pu_loader_nd, total_it)

    if epoch >= 0:
      # TODO (done): monitor second miniImagenet loss
      # validate
      # del ps_loader, ps_loader_second
      model.eval()
      acc = model.test_loop(val_loader, val_loader_nd, total_it)

    # save
      if acc > max_acc:
        print("best model! save...")
        max_acc = acc
        outfile = os.path.join(params.checkpoint_dir, 'best_model_{}.tar'.format(max_acc))
        model.save(outfile, epoch)
      else:
        print('GG!! best accuracy {:f}'.format(max_acc))
      if ((epoch + 1) % params.save_freq==0) or (epoch == stop_epoch - 1):
        outfile = os.path.join(params.checkpoint_dir, '{:d}.tar'.format(epoch + 1))
        model.save(outfile, epoch)
    else:
      print('--- Epoch {} done.'.format(epoch+1))

  return


# --- main function ---
if __name__=='__main__':

  # set numpy random seed
  np.random.seed(10)

  # parse argument
  params = parse_args('train')
  print('--- MAML training: {} ---\n'.format(params.name))
  print(params)

  # output and tensorboard dir
  params.tf_dir = '%s/log/%s'%(params.save_dir, params.name)
  params.checkpoint_dir = '%s/checkpoints/%s'%(params.save_dir, params.name)
  if not os.path.isdir(params.checkpoint_dir):
    os.makedirs(params.checkpoint_dir)

  # dataloader
  print('\n--- prepare dataloader ---')
  print('  train with multiple seen domains (unseen domain: {})'.format(params.testset))
  datasets = ['miniImagenet', 'cars', 'places', 'cub', 'plantae']
  datasets.remove(params.testset)
  #val_file = os.path.join(params.data_dir, 'miniImagenet', 'val.json')
  val_file = os.path.join(params.data_dir, params.testset, 'val.json')

  # model
  print('\n--- build MAML model ---')
  print('  train with model: %s'%params.model)
  if 'Conv' in params.model:
    image_size = 84
  else:
    image_size = 224

  n_query = max(1, int(16* params.test_n_way/params.train_n_way))
  train_few_shot_params   = dict(n_way = params.train_n_way, n_support = params.n_shot)
  base_datamgr            = SetDataManager(image_size, n_query = n_query, n_eposide = 100, **train_few_shot_params)
  test_few_shot_params    = dict(n_way = params.test_n_way, n_support = params.n_shot)
  val_datamgr             = SetDataManager(image_size, n_query = n_query, n_eposide = 100, **test_few_shot_params)
  val_loader              = val_datamgr.get_data_loader( val_file, aug = False)
  val_loader_nd           = val_datamgr.get_data_loader( val_file, aug = False)

  model = MAML(params, tf_path=params.tf_dir)
  model.cuda()

  # resume training
  start_epoch = params.start_epoch
  stop_epoch = params.stop_epoch
  if params.resume != '':
    resume_file = get_resume_file('%s/checkpoints/%s'%(params.save_dir, params.resume), params.resume_epoch)
    if resume_file is not None:
      start_epoch = model.resume(resume_file)
      print('  resume the training with at {} epoch (model file {})'.format(start_epoch, params.resume))
    else:
      raise ValueError('No resume file')
  # load pre-trained feature encoder
  else:
    if params.warmup == 'scratch':
      pass
    elif params.warmup == 'gg3b0':
      raise Exception('Must provide pre-trained feature-encoder file using --warmup option!')
    else:
      model.model.feature.load_state_dict(load_warmup_state('%s/checkpoints/%s'%(params.save_dir, params.warmup), params.method), strict=False)

  # training
  print('\n--- start the training ---')
  train(base_datamgr, datasets, val_loader, val_loader_nd, model, start_epoch, stop_epoch, params)
