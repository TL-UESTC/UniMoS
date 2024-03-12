# --------------------------------------------------------
# Reference from https://github.com/microsoft/Swin-Transformer
# --------------------------------------------------------

import os
import time

import yaml
from yacs.config import CfgNode as CN

_C = CN()

# Base config files
_C.BASE = ['']

# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
# Batch size for a single GPU, could be overwritten by command line argument
_C.DATA.BATCH_SIZE = 64
# Path to dataset, could be overwritten by command line argument
_C.DATA.DATA_PATH = ''
# Dataset name
_C.DATA.DATASET = 'imagenet'
# Input image size
_C.DATA.IMG_SIZE = 224
# Interpolation to resize image (random, bilinear, bicubic)
_C.DATA.INTERPOLATION = 'bicubic'
# Cache Data in Memory, could be overwritten by command line argument
_C.DATA.CACHE_MODE = 'part'
# Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.
_C.DATA.PIN_MEMORY = True
# Number of data loading threads
_C.DATA.NUM_WORKERS = 4
# Mean and standard
_C.DATA.MEAN = (0.485, 0.456, 0.406)
_C.DATA.STD = (0.229, 0.224, 0.225)

# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Model type
_C.MODEL.TYPE = 'swin'
# Model name
_C.MODEL.NAME = 'swin_base'
# Checkpoint to resume, could be overwritten by command line argument
_C.MODEL.RESUME = ''
# Number of classes, overwritten in data preparation
_C.MODEL.NUM_CLASSES = 31
# Dropout rate
_C.MODEL.DROP_RATE = 0.0
# Drop path rate
_C.MODEL.DROP_PATH_RATE = 0.
_C.MODEL.DIS = 'cosine'

# Swin Transformer parameters
_C.MODEL.SWIN = CN()
_C.MODEL.SWIN.PATCH_SIZE = 4
_C.MODEL.SWIN.IN_CHANS = 3
_C.MODEL.SWIN.EMBED_DIM = 128
_C.MODEL.SWIN.DEPTHS = [2, 2, 18, 2]
_C.MODEL.SWIN.NUM_HEADS = [4, 8, 16, 32]
_C.MODEL.SWIN.WINDOW_SIZE = 7
_C.MODEL.SWIN.MLP_RATIO = 4.
_C.MODEL.SWIN.QKV_BIAS = True
_C.MODEL.SWIN.QK_SCALE = None
_C.MODEL.SWIN.APE = False
_C.MODEL.SWIN.PATCH_NORM = True

# ViT parameters
_C.MODEL.VIT = CN()
_C.MODEL.VIT.EMBED_DIM = 786
_C.MODEL.VIT.PATCH_SIZE = 16
_C.MODEL.VIT.DEPTH = 12
_C.MODEL.VIT.NUM_HEADS = 12

# Deit parameters
_C.MODEL.DEIT = CN()
_C.MODEL.DEIT.EMBED_DIM = 768
_C.MODEL.DEIT.PATCH_SIZE = 16
_C.MODEL.DEIT.DEPTH = 12
_C.MODEL.DEIT.NUM_HEADS = 12

# Attention layer
_C.MODEL.ATTN = CN()
_C.MODEL.ATTN.FEATURE_DIM = 1024
_C.MODEL.ATTN.HEADS = 8
_C.MODEL.ATTN.HIDDEN_DIM = 2048

# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.EPOCHS = 20
_C.TRAIN.WEIGHT_DECAY = 0.05
_C.TRAIN.BIAS_WEIGHT_DECAY = 0.
_C.TRAIN.WARMUP_EPOCHS = 10
_C.TRAIN.WARMUP_LR_MULT = 0.1
_C.TRAIN.MIN_LR_MULT = 0.25
_C.TRAIN.BASE_LR = 5e-6
_C.TRAIN.LR_MULT = 10
_C.TRAIN.DECAY1 = 1.
_C.TRAIN.DECAY2 = 1.
_C.TRAIN.DECAY3 = 1.
# Clip gradient norm
_C.TRAIN.CLIP_GRAD = 5.0
# Gradient accumulation steps
# could be overwritten by command line argument
_C.TRAIN.ACCUMULATION_STEPS = 0
# Whether to use gradient checkpointing to save memory
# could be overwritten by command line argument
_C.TRAIN.USE_CHECKPOINT = False

# Optimizer
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = 'adamw'
# Optimizer Epsilon
_C.TRAIN.OPTIMIZER.EPS = 1e-8
# Optimizer Betas
_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)
# SGD momentum
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9
_C.TRAIN.R = 0.
_C.TRAIN.INTERVAL = 5
_C.TRAIN.WARMUP = 10

# -----------------------------------------------------------------------------
# Augmentation settings
# -----------------------------------------------------------------------------
_C.AUG = CN()
# Color jitter factor
_C.AUG.COLOR_JITTER = 0.4
# Use AutoAugment policy. "v0" or "original"
_C.AUG.AUTO_AUGMENT = 'rand-m9-mstd0.5-inc1'
# Random erase prob
_C.AUG.REPROB = 0.25
# Random erase mode
_C.AUG.REMODE = 'pixel'
# Random erase count
_C.AUG.RECOUNT = 1

# -----------------------------------------------------------------------------
# Testing settings
# -----------------------------------------------------------------------------
_C.TEST = CN()
# Whether to use center crop when testing
_C.TEST.CROP = True

# -----------------------------------------------------------------------------
# Hyper-parameters
# -----------------------------------------------------------------------------
_C.hyper_parameters = CN()
# Whether to use center crop when testing
_C.hyper_parameters.alpha = 0.8
_C.hyper_parameters.beta = 3

# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------
# Mixed precision opt level, if O0, no amp is used ('O0', 'O1', 'O2')
# overwritten by command line argument
_C.AMP_OPT_LEVEL = ''
# Path to output folder, overwritten by command line argument
_C.OUTPUT = ''
# Tag of experiment, overwritten by command line argument
_C.TAG = 'default'
# Frequency to save checkpoint
_C.SAVE_FREQ = 25
# Frequency to logging info
_C.PRINT_FREQ = 10000
# Fixed random seed
_C.SEED = 123
# Perform evaluation only, overwritten by command line argument
_C.EVAL_MODE = False
# Test throughput only, overwritten by command line argument
_C.THROUGHPUT_MODE = False
# local rank for DistributedDataParallel, given by command line argument
_C.LOCAL_RANK = 0
_C.RETURN_PATH = False
_C.VAL = False
_C.VOTE_NUM = 4


def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    print('=> merge config from {}'.format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()


def update_config(config, args):
    _update_config_from_file(config, args.cfg)

    config.defrost()

    # merge from specific arguments
    if args.batch_size:
        config.DATA.BATCH_SIZE = args.batch_size
    if args.data_root_path:
        config.DATA.DATA_PATH = args.data_root_path + args.dataset
    if args.source:
        config.DATA.SOURCE = args.source
    if args.target:
        config.DATA.TARGET = args.target
    if args.output:
        config.OUTPUT = args.output

    if args.dataset:
        config.DATA.DATASET = args.dataset

    if config.DATA.DATASET == 'office31':
        config.MODEL.NUM_CLASSES = 31
    elif config.DATA.DATASET == 'domainnet':
        config.MODEL.NUM_CLASSES = 345
    elif config.DATA.DATASET == 'office_home':
        config.MODEL.NUM_CLASSES = 65
    elif config.DATA.DATASET == 'visda':
        config.MODEL.NUM_CLASSES = 12

    config.SEED = args.seed

    try:
        if args.lr != 0:
            config.TRAIN.BASE_LR = args.lr
        if args.decay1 != 0:
            config.TRAIN.DECAY1 = args.decay1
        if args.decay2 != 0:
            config.TRAIN.DECAY2 = args.decay2
    except:
        print("Config updated")

    config.freeze()


def get_config(args):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    update_config(config, args)

    return config
