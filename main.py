import argparse
import numpy as np
import torch.backends.cudnn as cudnn
from configs.config import get_config
from model import *
from utils.build import *
from utils.logger import *
#warnings.filterwarnings('ignore')

def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str,
                        default="configs/unimos.yaml",
                        metavar="FILE", help='path to config file', )
    parser.add_argument('--batch_size', type=int, default=32, help="batch size for single GPU")
    parser.add_argument('--devices', type=int, default=0, help="device IDs")
    parser.add_argument('--dataset', type=str, default='office_home', choices=['office_home'], help='dataset used')
    parser.add_argument('--data-root-path', type=str, default='dataset/', help='path to dataset txt files')
    parser.add_argument('--source', type=str, default='Art', help='source name', choices=['Art','Clipart','Product','Real_World'])
    parser.add_argument('--target', type=str, default='Clipart', help='target name', choices=['Art','Clipart','Product','Real_World'])
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--output', default='log', type=str, metavar='PATH')
    parser.add_argument('--log', default='log/', help='log path')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--file', default=None, type=str)
    parser.add_argument('--lr', default=0.003, type=float)
    parser.add_argument('--seed', default=2023, type=int)
    parser.add_argument('--model', default='RN50', type=str)    
    parser.add_argument('--momen', default=0.99, type=float)   
    parser.add_argument('--tau', default=0.5, type=float)    
    parser.add_argument('--l', default=None, type=float)    
    parser.add_argument('--alpha_reg', default=0.01, type=float)    
    parser.add_argument('--alpha_srctxt', default=1, type=float)    
    parser.add_argument('--alpha_srcvis', default=1, type=float)    
    parser.add_argument('--end', default=0.3, type=float)    

    args = parser.parse_args()
    if args.file is None:
        args.file = str(args.seed)

    args.output += '/{}/{}/{}_{}'.format(args.dataset, args.model, args.source, args.target)
    
    config = get_config(args)

    if args.dataset == 'office_home':
        args.class_num = 65
    else:
        raise NotImplementedError()

    return args, config


def print_args(args):
    strr = '\n'
    for k in vars(args):
        strr += '{}: {}\n'.format(k, getattr(args, k))
    return strr


if __name__ == '__main__':
    args, config = parse_option()
    torch.cuda.set_device(args.devices)
    seed = config.SEED 
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    cudnn.deterministic = True
    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, name=f"{config.MODEL.NAME}", file=args.file)
    logger.info(print_args(args))
    args.logger = logger
    path = os.path.join(config.OUTPUT, "config_{}.yaml".format(args.file))
    if args.dataset in ['office_home'] :
        args.load_data_method = data_load
    else:
        raise NotImplementedError()

    trainer = UniMoS(args, config)
    trainer.train()
