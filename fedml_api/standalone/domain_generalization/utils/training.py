import copy

import torch
from argparse import Namespace
from fedml_api.standalone.domain_generalization.models.utils.federated_model import FederatedModel
from fedml_api.standalone.domain_generalization.datasets.utils.federated_dataset import FederatedDataset
from typing import Tuple
from torch.utils.data import DataLoader
import numpy as np
from fedml_api.standalone.domain_generalization.utils.logger import CsvWriter
from collections import Counter

import wandb


def global_evaluate(model: FederatedModel, test_dl: DataLoader, setting: str, name: str) -> Tuple[list, list]:
    accs = []
    net = model.global_net
    status = net.training
    net.eval()
    for j, dl in enumerate(test_dl):
        correct, total, top1, top5 = 0.0, 0.0, 0.0, 0.0
        for batch_idx, (images, labels) in enumerate(dl):
            with torch.no_grad():
                images, labels = images.to(model.device), labels.to(model.device)
                if model.NAME == 'nefl':
                    outputs, _ = net(images)
                else:
                    outputs = net(images)
                _, max5 = torch.topk(outputs, 5, dim=-1)
                labels = labels.view(-1, 1)
                top1 += (labels == max5[:, 0:1]).sum().item()
                top5 += (labels == max5).sum().item()
                total += labels.size(0)
        top1acc = round(100 * top1 / total, 2)
        top5acc = round(100 * top5 / total, 2)
        accs.append(top1acc)
    net.train(status)
    return accs

def local_evaluate(model: FederatedModel, test_dl: DataLoader, domains_list: list, selected_domain_list: list, setting: str, name: str) -> list:
    all_accs = {}
    for i, net in enumerate(model.nets_list):
        status = net.training
        net.eval()
        domain = selected_domain_list[i]
        domain_index = domains_list.index(domain)
        if domain_index not in all_accs.keys():
            all_accs[domain_index] = []
        correct, total, top1, top5 = 0.0, 0.0, 0.0, 0.0
        for batch_idx, (images, labels) in enumerate(test_dl[domain_index]):
            with torch.no_grad():
                images, labels = images.to(model.device), labels.to(model.device)
                outputs = net(images)
                _, max5 = torch.topk(outputs, 5, dim=-1)
                labels = labels.view(-1, 1)
                top1 += (labels == max5[:, 0:1]).sum().item()
                top5 += (labels == max5).sum().item()
                total += labels.size(0)
        top1acc = round(100 * top1 / total, 2)
        all_accs[domain_index].append(top1acc)
        net.train(status)

    avg_accs = []
    for i in range(len(all_accs)):
        avg_acc = round(sum(all_accs[i]) / len(all_accs[i]), 2)
        avg_accs.append(avg_acc)
    return avg_accs

def train(model: FederatedModel, private_dataset: FederatedDataset,
          args: Namespace) -> None:
    if args.csv_log:
        csv_writer = CsvWriter(args, private_dataset)

    model.N_CLASS = private_dataset.N_CLASS
    domains_list = private_dataset.DOMAINS_LIST
    domains_len = len(domains_list)

    if args.rand_dataset:
        max_num = 10
        is_ok = False

        while not is_ok:
            if model.args.dataset == 'fl_officecaltech':
                selected_domain_list = np.random.choice(domains_list, size=args.parti_num - domains_len, replace=True, p=None)
                selected_domain_list = list(selected_domain_list) + domains_list
            elif model.args.dataset == 'fl_digits':
                # selected_domain_list = np.random.choice(domains_list, size=args.parti_num, replace=True, p=None)
                selected_domain_list = np.random.choice(domains_list, size=args.parti_num - domains_len, replace=True, p=None)
                selected_domain_list = list(selected_domain_list) + domains_list

            result = dict(Counter(selected_domain_list))

            for k in result:
                if result[k] > max_num:
                    is_ok = False
                    break
            else:
                is_ok = True

    else:
        selected_domain_dict = {'mnist': 6, 'usps': 4, 'svhn': 3, 'syn': 7}

        selected_domain_list = []
        for k in selected_domain_dict:
            domain_num = selected_domain_dict[k]
            for i in range(domain_num):
                selected_domain_list.append(k)

        selected_domain_list = np.random.permutation(selected_domain_list)

        result = Counter(selected_domain_list)
    print(result)

    print(selected_domain_list)
    pri_train_loaders, test_loaders = private_dataset.get_data_loaders(selected_domain_list)
    model.trainloaders = pri_train_loaders
    if hasattr(model, 'ini'):
        model.ini()

    accs_dict = {}
    mean_accs_list = []
    best_acc = 0
    best_accs = []

    Epoch = args.communication_epoch
    for epoch_index in range(Epoch):
        model.epoch_index = epoch_index

        if hasattr(model, 'loc_update'):
            epoch_loc_loss_dict = model.loc_update(pri_train_loaders)

        if args.model in ['localtest']:
            accs = local_evaluate(model, test_loaders, domains_list, selected_domain_list, private_dataset.SETTING, private_dataset.NAME)
            model.aggregate_nets()
        else:
            accs = global_evaluate(model, test_loaders, private_dataset.SETTING, private_dataset.NAME)


        mean_acc = round(np.mean(accs, axis=0), 3)
        mean_accs_list.append(mean_acc)
        for i in range(len(accs)):
            if i in accs_dict:
                accs_dict[i].append(accs[i])
            else:
                accs_dict[i] = [accs[i]]

        if mean_acc > best_acc:
            best_acc = mean_acc
        if len(best_accs) == 0:
            best_accs = copy.deepcopy(accs)
        for i in range(len(accs)):
            if accs[i] > best_accs[i]:
                best_accs[i] = accs[i]

        if args.wandb:
            wandb.log({"Best_Acc": best_acc, "Mean_Acc": mean_acc, "round": epoch_index})
            if len(best_accs) == 0:
                best_accs = copy.deepcopy(accs)
            for i in range(len(accs)):
                name = "Domain"+str(i)
                wandb.log({name+"_Acc": accs[i], name+"_BestAcc": best_accs[i], "round": epoch_index})

        print('Round:', str(epoch_index), 'Method:', model.args.model,
              'Mean_Acc:', str(mean_acc), 'Best_Acc:', str(best_acc))
        print('Domain_Acc:', accs, 'Domain_BestAcc:', best_accs)

    if args.csv_log:
        csv_writer.write_acc(accs_dict, mean_accs_list)
