import datetime
import os
import math
from tqdm import tqdm

import random
import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset
import torch.distributed as distributed

from model.trainer import XMemTrainer
from dataset.static_dataset import StaticTransformDataset
from dataset.vos_dataset import VOSDataset

from util.logger import TensorboardLogger
from util.configuration import Configuration
from util.load_subset import load_sub_davis, load_sub_yv, _load_sub


"""
Initial setup
"""
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

# Init distributed environment
distributed.init_process_group(backend="nccl")
print(f'CUDA Device count: {torch.cuda.device_count()}')

# Parse command line arguments
raw_config = Configuration()
raw_config.parse()

torch.backends.cudnn.enabled = False
if raw_config['benchmark']:
    torch.backends.cudnn.benchmark = True

# Get current git info
# repo = git.Repo(".")
# git_info = str(repo.active_branch)+' '+str(repo.head.commit.hexsha)
git_info = ''

local_rank = torch.distributed.get_rank()
world_size = torch.distributed.get_world_size()
torch.cuda.set_device(local_rank)

print(f'I am rank {local_rank} in this world of size {world_size}!')

network_in_memory = None
stages = raw_config['stages']
stages_to_perform = list(stages)
for si, stage in enumerate(stages_to_perform):

    # Set seed to ensure the same initialization
    torch.manual_seed(14159265)
    np.random.seed(14159265)
    random.seed(14159265)

    # Pick stage specific hyperparameters out
    stage_config = raw_config.get_stage_parameters(stage)
    config = dict(**raw_config.args, **stage_config)
    if config['exp_id'] != 'NULL':
        config['exp_id'] = config['exp_id']+'_s%s'%stages[:si+1]

    config['single_object'] = (stage == '0')

    config['num_gpus'] = world_size
    if config['batch_size']//config['num_gpus']*config['num_gpus'] != config['batch_size']:
        raise ValueError('Batch size must be divisible by the number of GPUs.')
    config['batch_size'] //= config['num_gpus']
    config['num_workers'] //= config['num_gpus']
    print(f'We are assuming {config["num_gpus"]} GPUs.')

    print(f'We are now starting stage {stage}')

    """
    Model related
    """
    if local_rank == 0:
        # Logging
        if config['exp_id'].lower() != 'null':
            print('I will take the role of logging!')
            long_id = '%s_%s' % (datetime.datetime.now().strftime('%b%d_%H.%M.%S'), config['exp_id'])
        else:
            long_id = None
        logger = TensorboardLogger(config['exp_id'], long_id, git_info)
        logger.log_string('hyperpara', str(config))

        # Construct the rank 0 model
        model = XMemTrainer(config, logger=logger, 
                        save_path=os.path.join('saves', long_id, long_id) if long_id is not None else None, 
                        local_rank=local_rank, world_size=world_size).train()
    else:
        # Construct model for other ranks
        model = XMemTrainer(config, local_rank=local_rank, world_size=world_size).train()

    # Load pertrained model if needed
    if raw_config['load_checkpoint'] is not None:
        total_iter = model.load_checkpoint(raw_config['load_checkpoint'])
        raw_config['load_checkpoint'] = None
        print('Previously trained model loaded!')
    else:
        total_iter = 0

    if network_in_memory is not None:
        print('I am loading network from the previous stage')
        model.load_network_in_memory(network_in_memory)
        network_in_memory = None
    elif raw_config['load_network'] is not None:
        print('I am loading network from a disk, as listed in configuration')
        model.load_network(raw_config['load_network'])
        raw_config['load_network'] = None

    """
    Dataloader related
    """
    # To re-seed the randomness everytime we start a worker
    def worker_init_fn(worker_id): 
        worker_seed = torch.initial_seed()%(2**31) + worker_id + local_rank*100
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def construct_loader(dataset, bs=config['batch_size']):
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, rank=local_rank, shuffle=True)
        train_loader = DataLoader(dataset, bs, sampler=train_sampler, num_workers=config['num_workers'],
                                worker_init_fn=worker_init_fn, drop_last=True)
        return train_sampler, train_loader

    def renew_vos_loader(max_skip, finetune=False):
        # //5 because we only have annotation for every five frames

        ct_dataset = VOSDataset('/data/yanzhn/Datasets/vos_AMOS_CT/image',
                                '/data/yanzhn/Datasets/vos_AMOS_CT/mask',
                                max_skip//5, type='ct',
                                subset=_load_sub('/data/yanzhn/Datasets/vos_AMOS_CT/train.txt'), num_frames=config['num_frames'], finetune=finetune)
        
        mri_dataset = VOSDataset('/data/yanzhn/Datasets/vos_AMOS_MRI/image',
                                 '/data/yanzhn/Datasets/vos_AMOS_MRI/mask',
                                 max_skip//5, type='mri',
                                 subset=_load_sub('/data/yanzhn/Datasets/vos_AMOS_MRI/train.txt'), num_frames=config['num_frames'], finetune=finetune)
        
        endo_dataset = VOSDataset('/data/yanzhn/Datasets/vos_CholecSeg8K_Endoscope/image',
                                  '/data/yanzhn/Datasets/vos_CholecSeg8K_Endoscope/mask',
                                  max_skip//5, type='endo',
                                  subset=_load_sub('/data/yanzhn/Datasets/vos_CholecSeg8K_Endoscope/train.txt'), num_frames=config['num_frames'], finetune=finetune)
        
        em_dataset = VOSDataset('/data/yanzhn/Datasets/vos_EM/image',
                                '/data/yanzhn/Datasets/vos_EM/mask',
                                max_skip//5, type='em',
                                subset=_load_sub('/data/yanzhn/Datasets/vos_EM/train.txt'), num_frames=config['num_frames'], finetune=finetune)
        
        us_dataset = VOSDataset('/data/yanzhn/Datasets/vos_CAMUS/image',
                                '/data/yanzhn/Datasets/vos_CAMUS/mask',
                                max_skip//5, type='us',
                                subset=_load_sub('/data/yanzhn/Datasets/vos_CAMUS/train.txt'), num_frames=config['num_frames'], finetune=finetune)
        
        train_dataset = ConcatDataset([ct_dataset, mri_dataset, endo_dataset, em_dataset, us_dataset])
        print(f'Concat dataset size: {len(train_dataset)}')
        print(f'Renewed with {max_skip=}')

        return construct_loader(train_dataset)
    

    def renew_vos_loader_valid(max_skip, finetune=False):
        # //5 because we only have annotation for every five frames
        ct_dataset = VOSDataset('/data/yanzhn/Datasets/vos_AMOS_CT/image',
                                '/data/yanzhn/Datasets/vos_AMOS_CT/mask',
                                max_skip//5, type='ct',
                                subset=_load_sub('/data/yanzhn/Datasets/vos_AMOS_CT/val.txt'), num_frames=config['num_frames'], finetune=finetune)
        
        mri_dataset = VOSDataset('/data/yanzhn/Datasets/vos_AMOS_MRI/image',
                                 '/data/yanzhn/Datasets/vos_AMOS_MRI/mask',
                                 max_skip//5, type='mri',
                                 subset=_load_sub('/data/yanzhn/Datasets/vos_AMOS_MRI/val.txt'), num_frames=config['num_frames'], finetune=finetune)
        
        endo_dataset = VOSDataset('/data/yanzhn/Datasets/vos_CholecSeg8K_Endoscope/image',
                                  '/data/yanzhn/Datasets/vos_CholecSeg8K_Endoscope/mask',
                                  max_skip//5, type='endo',
                                  subset=_load_sub('/data/yanzhn/Datasets/vos_CholecSeg8K_Endoscope/val.txt'), num_frames=config['num_frames'], finetune=finetune)
        
        em_dataset = VOSDataset('/data/yanzhn/Datasets/vos_EM/image',
                                '/data/yanzhn/Datasets/vos_EM/mask',
                                max_skip//5, type='em',
                                subset=_load_sub('/data/yanzhn/Datasets/vos_EM/val.txt'), num_frames=config['num_frames'], finetune=finetune)
        
        us_dataset = VOSDataset('/data/yanzhn/Datasets/vos_CAMUS/image',
                                '/data/yanzhn/Datasets/vos_CAMUS/mask',
                                max_skip//5, type='us',
                                subset=_load_sub('/data/yanzhn/Datasets/vos_CAMUS/val.txt'), num_frames=config['num_frames'], finetune=finetune)
        
        valid_dataset = ConcatDataset([ct_dataset, mri_dataset, endo_dataset, em_dataset, us_dataset])
        print(f'valid Concat dataset size: {len(valid_dataset)}')
        print(f'valid Renewed with {max_skip=}')

        return construct_loader(valid_dataset, 16)

    """
    Dataset related
    """

    """
    These define the training schedule of the distance between frames
    We will switch to max_skip_values[i] once we pass the percentage specified by increase_skip_fraction[i]
    Not effective for stage 0 training
    The initial value is not listed here but in renew_vos_loader(X)
    """
    max_skip_values = [10, 15, 5, 5]

    if stage == '0':
        static_root = os.path.expanduser(config['static_root'])
        # format: path, method (style of storing images), mutliplier
        train_dataset = StaticTransformDataset(
            [
                (os.path.join(static_root, 'fss'), 0, 1),
                (os.path.join(static_root, 'DUTS-TR'), 1, 1),
                (os.path.join(static_root, 'DUTS-TE'), 1, 1),
                (os.path.join(static_root, 'ecssd'), 1, 1),
                (os.path.join(static_root, 'BIG_small'), 1, 5),
                (os.path.join(static_root, 'HRSOD_small'), 1, 5),
            ], num_frames=config['num_frames'])
        train_sampler, train_loader = construct_loader(train_dataset)

        print(f'Static dataset size: {len(train_dataset)}')
    elif stage == '1':
        increase_skip_fraction = [0.1, 0.3, 0.8, 100]
        bl_root = os.path.join(os.path.expanduser(config['bl_root']))

        train_sampler, train_loader = renew_bl_loader(5)
        renew_loader = renew_bl_loader
    else:
        # stage 2 or 3
        increase_skip_fraction = [0.1, 0.3, 0.9, 100]
        # VOS dataset, 480p is used for both datasets
        yv_root = os.path.join(os.path.expanduser(config['yv_root']), 'train_480p')
        davis_root = os.path.join(os.path.expanduser(config['davis_root']), '2017', 'trainval')

        train_sampler, train_loader = renew_vos_loader(5)
        renew_loader = renew_vos_loader
        
        valid_sampler, valid_loader = renew_vos_loader_valid(5)
        renew_loader_valid = renew_vos_loader_valid



    """
    Determine max epoch
    """
    total_epoch = math.ceil(config['iterations']/len(train_loader))
    current_epoch = total_iter // len(train_loader)
    print(f'We approximately use {total_epoch} epochs.')
    if stage != '0':
        change_skip_iter = [round(config['iterations']*f) for f in increase_skip_fraction]
        # Skip will only change after an epoch, not in the middle
        print(f'The skip value will change approximately at the following iterations: {change_skip_iter[:-1]}')

    """
    Starts training
    """
    finetuning = False
    # Need this to select random bases in different workers
    np.random.seed(np.random.randint(2**30-1) + local_rank*100)
    try:
        while total_iter < config['iterations'] + config['finetune']:
            
            # Crucial for randomness! 
            train_sampler.set_epoch(current_epoch)
            current_epoch += 1
            print(f'Current epoch: {current_epoch}')
            
            # model.save_network(total_iter)

            # Train loop
            model.train()
            for data in train_loader:
                # Update skip if needed
                if stage!='0' and total_iter >= change_skip_iter[0]:
                    while total_iter >= change_skip_iter[0]:
                        cur_skip = max_skip_values[0]
                        max_skip_values = max_skip_values[1:]
                        change_skip_iter = change_skip_iter[1:]
                    print(f'Changing skip to {cur_skip=}')
                    train_sampler, train_loader = renew_loader(cur_skip)
                    break

                # fine-tune means fewer augmentations to train the sensory memory
                if config['finetune'] > 0 and not finetuning and total_iter >= config['iterations']:
                    train_sampler, train_loader = renew_loader(cur_skip, finetune=True)
                    finetuning = True
                    model.save_network_interval = 1000
                    print('finetune')
                    break

                model.do_pass(data, total_iter)
                total_iter += 1

                if total_iter >= config['iterations'] + config['finetune']:
                    break
            
    finally:
        if not config['debug'] and model.logger is not None and total_iter>5000:
            model.save_network(total_iter)
            model.save_checkpoint(total_iter)

    network_in_memory = model.XMem.module.state_dict()

distributed.destroy_process_group()
