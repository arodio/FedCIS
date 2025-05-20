from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import pickle
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
from loguru import logger
from torch.nn.modules.batchnorm import BatchNorm2d

from models.cnn import CNNEE
from models.resnets import ResNetEE


class ComputeNode():
    def __init__(self, nEE, epoch, loader, args, client_id):
        self.args = args
        self.EE_counter = -1
        self.nEE = nEE
        self.sampled_probability = self.args.sampled_probability[nEE - 1]
        self.client_id = client_id
        self.node_name = f'Node_{client_id}_'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.args.seed:
            torch.manual_seed(self.args.seed)
            np.random.seed(self.args.seed)
        if args.arch == 'resnet':
            self.model = ResNetEE(self.args.layers_widths, nClassifiers=args.nClassifiers, growth=self.args.resnet_growthRate,
                                  channels=self.args.resnet_nChannels, num_classes=args.num_classes,
                                  init_res=False, layers_per_classifier=args.layers_per_classifier)
            logger.debug(len(self.model.classifiers))
            self.model.set_nBlocks(nEE)

            if self.args.no_batch_norm:
                print("no batch norm")

                def replace_bn_with_identity(model):
                    for child_name, child in model.named_children():
                        if isinstance(child, nn.BatchNorm2d):
                            setattr(model, child_name, nn.Identity())
                        else:
                            replace_bn_with_identity(child)

                replace_bn_with_identity(self.model)

        elif args.arch == 'cnn':
            logger.debug(args.num_classes)
            self.model = CNNEE(model_config=list(args.layers_per_classifier), nClassifiers=args.nClassifiers)
        if args.float64:
            self.model = self.model.double()
        self.model = self.model.cpu()
        self.optimizer = torch.optim.SGD(self.model.parameters(), self.args.lr,
                                         momentum=self.args.momentum,
                                         weight_decay=self.args.weight_decay)
        for param_group in self.optimizer.param_groups:
            param_group['initial_lr'] = self.args.lr
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.kl_loss = nn.KLDivLoss(reduction='batchmean').to(self.device)
        self.epoch = epoch
        # self.root_data_loader = loader.get_data(rank)
        self.train_loader, self.val_loader, self.test_loader = loader.get_data(client_id)
        if args.lr_type == 'multistep':
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                                  milestones=args.lr_milestones, gamma=0.1, )
        elif args.lr_type == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, eta_min=args.lr * (0.1) ** 2,
                                                                        T_max=args.epochs)
        elif args.lr_type == 'linear':
            self.scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer, T_max=args.epochs)
        elif args.lr_type == 'fixed':
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[+np.inf], gamma=1)
        elif args.lr_type == 'none':
            self.scheduler = None
        else:
            raise Exception('Not Implemented')

        self.EE_percentage = np.ones(self.model.nClassifiers) * -1  # .to(device)
        self.ConfidenceData = np.ones(self.model.nClassifiers) * -1
        self.classifiers_weight = np.ones(self.model.nClassifiers)
        self.train_stat = None
        self.active = False
        if self.args.single_batch_test:
            self.dataset_size = len(self.train_loader.sampler.data_source)
        else:
            self.dataset_size = len(self.train_loader.sampler.indices)
        self.earlyexit_weights_validation = np.zeros(self.args.nClassifiers)
        self.thinned_loss_count = np.zeros(self.args.nClassifiers)
        self.thinned_loss = np.zeros(self.args.nClassifiers)
        self.thinned_conf = np.zeros(self.args.nClassifiers)

    def move_model_to_cpu(self):
        self.model.to('cpu')

    def move_model_to_gpu(self):
        self.model.to(self.device)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0


def accuracy(output, target, topk=(1, 3)):
    """Computes the precor@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        # correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
