import os
import sys
import socket
import time
import uuid
import datetime
import setproctitle
import torch
import torch.multiprocessing
import wandb
import warnings
from fedml_api.standalone.domain_generalization.datasets import Priv_NAMES as DATASET_NAMES
from fedml_api.standalone.domain_generalization.models import get_all_models
from argparse import ArgumentParser
from fedml_api.standalone.domain_generalization.utils.args import add_management_args
from fedml_api.standalone.domain_generalization.datasets import get_prive_dataset
from fedml_api.standalone.domain_generalization.models import get_model
from fedml_api.standalone.domain_generalization.utils.training import train
from fedml_api.standalone.domain_generalization.utils.best_args import best_args
from fedml_api.standalone.domain_generalization.utils.conf import set_random_seed

torch.multiprocessing.set_sharing_strategy('file_system')
warnings.filterwarnings("ignore")
conf_path = os.getcwd()
sys.path.append('../../..')
sys.path.append(conf_path)
sys.path.append(conf_path + '/datasets')
sys.path.append(conf_path + '/backbone')
sys.path.append(conf_path + '/models')



def parse_args():
    parser = ArgumentParser(description='You Only Need Me')

    parser.add_argument('-pf', '--prefix', type=str, default='', metavar='PFX',
                        help='dataset prefix for logging & checkpoint saving')
    parser.add_argument('--communication_epoch', type=int, default=100, help='Total communication rounds of Federated Learning.')
    parser.add_argument('--local_epoch', type=int, default=5, help='Local epochs for local model updating.')
    parser.add_argument('--parti_num', type=int, default=10, help='Number of participants.')
    parser.add_argument('--model', type=str, default='dapperfl', help='Name of FL framework.',
                        choices=get_all_models())
    parser.add_argument('--dataset', type=str, default='fl_officecaltech',  # fl_officecaltech fl_digits
                        choices=DATASET_NAMES, help='Datasets used in the experiment.')
    parser.add_argument('--pr_strategy', type=str, default='AD', help='Model pruning strategy.')
    parser.add_argument('-bb', '--backbone', type=str, default='res18',
                        help='Backbone global model.')
    parser.add_argument('-a', '--alpha', type=float, default=0.9, help='Coefficient alpha in co-pruning')
    parser.add_argument('-amin', '--alpha_min', type=float, default=0.1, help='Coefficient alpha_min in co-pruning')
    parser.add_argument('-e', '--epsilon', type=float, default=0.2, help='Coefficient epsilon in co-pruning')
    parser.add_argument('-reg', '--reg_coeff', type=float, default=1e-2, help='Coefficient for L2 regularization')


    parser.add_argument('-wb', '--wandb', type=int, default=1, help='Enable wandb.')
    parser.add_argument('--device_id', type=int, default=0, help='The Device Id for Experiment')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed.')
    parser.add_argument('--rand_dataset', type=int, default=1, help='The random dataset.')
    parser.add_argument('--learning_decay', type=bool, default=False, help='The Option for Learning Rate Decay')
    parser.add_argument('--averaing', type=str, default='weight', help='The Option for averaging strategy')
    parser.add_argument('--online_ratio', type=float, default=1, help='The Ratio for Online Clients')


    torch.set_num_threads(8)
    add_management_args(parser)
    args = parser.parse_args()
    best = best_args[args.dataset][args.model]

    for key, value in best.items():
        setattr(args, key, value)
    if args.seed is not None:
        set_random_seed(args.seed)

    return args


def main(args=None):
    if args is None:
        args = parse_args()

    priv_dataset = get_prive_dataset(args)
    backbones_list = priv_dataset.get_backbone(args.parti_num, args.backbone)
    model = get_model(backbones_list, args, priv_dataset.get_transform())
    args.arch = model.nets_list[0].name
    if args.wandb:
        prefix = ''
        if args.prefix != '':
            prefix = args.prefix + '-'
        wandb.init(
            project="feddg",
            name=prefix+str(args.model) + "-" + str(args.dataset),
            config=args
        )
    print(args)

    formatted_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(formatted_time)
    setproctitle.setproctitle('{}_{}_{}_{}_{}'.format(args.model, args.parti_num, args.dataset, args.communication_epoch, args.local_epoch))

    train(model, priv_dataset, args)

    formatted_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(formatted_time)

    if args.wandb:
        wandb.finish()


if __name__ == '__main__':
    main()
