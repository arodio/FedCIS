import os
import pickle
import random
import shutil
import time
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from torch import nn

from args import arg_parser
from fedclients.computeNode import ComputeNode, AverageMeter, accuracy
from models.resnets import ResNetEE
from fvcore.nn import FlopCountAnalysis


def load_args():
    args = arg_parser.parse_args()
    # Reproducibility code
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)
    # Always attempt to resume training
    args.resume = True
    # args.topology_depth = args.n_clients - 1 if args.topology_depth >= args.n_clients else args.topology_depth
    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    args.layers_widths = np.array(list(map(int, args.resnet_layers_widths.split('-'))))
    args.lr_milestones = np.array(list(map(int, args.lr_milestones.split('-'))))
    args.layers_per_classifier = np.array(list(map(int, args.resnet_layers_per_classifier.split('-'))))
    if args.sampled_reverse:
        args.sampled_probability = list(
            map(float, args.sampled_probability_per_layer.split('-'))) + [0.]  # Add 0. for last exit
    else:
        args.sampled_probability = [0.] + list(
            map(float, args.sampled_probability_per_layer.split('-')))  # Add 0. for the first exit
    layer_m_1 = -1
    for layer in args.layers_per_classifier:
        if layer <= layer_m_1 and layer_m_1 != -1:
            raise Exception(
                'Layers assigned to a classifier should be strictly increasing (e.g., 2-4-5, '
                '2 layers assigned to classifier 1, '
                '4 layers assigned to classifier 2, '
                '5 layers assigned to classifier 3).')
        layer_m_1 = layer

    if 'mnist' in args.data:
        args.in_channels = 1
    else:
        args.in_channels = 3

    if args.data == 'cifar10' or args.data == 'mnist':
        args.num_classes = 10

    elif args.data.startswith('cifar100'):
        args.num_classes = 100

    elif args.data == 'emnist':
        args.num_classes = 62
    else:
        args.num_classes = 1000

    if not os.path.exists(args.save):
        os.makedirs(args.save, exist_ok=True)

    return args


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = [AverageMeter() for _ in range(model.nClassifiers)]
    top1 = [AverageMeter() for _ in range(model.nClassifiers)]
    model.eval()
    end = time.time()
    # logger.debug(f'============================================================')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # accuracy = Accuracy(top_k=1).to(device)
    indx = []
    acc = []
    with torch.no_grad():
        for i, (input, target_index) in enumerate(val_loader):
            # model.apply(deactivate_batchnorm)
            target = target_index[:, 0].type(torch.long).to(device)
            input = input.to(device).to(device)
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)
            data_time.update(time.time() - end)
            output = model(input_var)
            if not isinstance(output, list):
                output = [output]
            # output = [output[-1]]
            loss = 0.0
            for j in range(len(output)):
                loss += criterion(output[j], target_var)
                losses[j].update(loss.item(), input.size(0))
            for j in range(len(output)):
                prec1, prec3 = accuracy(output[j].data, target, topk=(1, 3))
                top1[j].update(prec1.item(), input.size(0))
            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            indx.extend(target_index[:, 1].cpu().numpy())
            acc.append(prec1.item())

    return losses, top1


def validate_thinned(test_loader, model, criterion, args, served_mask):
    batch_time = AverageMeter()
    losses = AverageMeter()
    data_time = AverageMeter()
    top1, top3 = AverageMeter(), AverageMeter()
    end = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    _ind = []
    total_rate = 0
    with torch.no_grad():
        for i, (input, target_index) in enumerate(test_loader):
            target = target_index[:, 0].type(torch.long).to(device)
            mask = served_mask[target_index[:, 1].cpu().numpy()]
            _ind.extend(target_index[:, 1].cpu().numpy())
            if sum(mask) == 0:
                continue
            target = target[mask.astype(np.bool)]
            input = input[mask.astype(np.bool)]
            mask = mask[mask != 0]
            input = input.to(device).to(device)
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)
            data_time.update(time.time() - end)
            output = model(input_var)
            if not isinstance(output, list):
                output = [output]
            loss = 0.0
            # inflated_indexes = []
            # for i in np.where(mask != 0)[0]:
            #     inflated_indexes.extend([i] * int(mask[i]))
            # inflated_indexes = (inflated_indexes)
            for i in range(len(mask)):
                loss += criterion(output[-1][i], target_var[i]) * mask[i]
                losses.update(loss.item(), 1)
                prec1, prec3 = accuracy(output[-1][[i]].data, target[[i]], topk=(1, 3))
                top1.update(prec1.item() * mask[i], 1)
                top3.update(prec3.item() * mask[i], 1)
                total_rate += mask[i]
            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
    logger.debug(np.unique(_ind).size)
    return losses.avg, top1, total_rate


class NodeStats:
    def __init__(self, computeNode):
        self.computeNode = computeNode
        self.E = self.computeNode.model.nClassifiers
        score_header = 'Timestamp\tEpoch\tTask Type\tlr'

        for e in range(self.E):
            score_header += f'\tTrain Loss[{e}]'
        for e in range(self.E):
            score_header += f'\tTrain Precision[{e}]'
        for e in range(self.E):
            score_header += f'\tValidation Loss[{e}]'
        for e in range(self.E):
            score_header += f'\tValidation Precision[{e}]'
        for e in range(self.E):
            score_header += f'\tTest Loss[{e}]'
        for e in range(self.E):
            score_header += f'\tTest Precision[{e}]'

        score_header += '\tLoss on Partial Data'
        score_header += f'\tTest Precision on Partial Data (%)'
        score_header += f'\tPartial Rate'
        score_header += f'\tPartial Count (Support Cardinality)'
        self.score_header = score_header
        self.best_val_prec1 = 0
        self.computeNode.nEE = self.computeNode.nEE

    def save_stats(self, epoch, full_test_loader):
        logger.debug(self.computeNode.node_name)
        test_scores = []
        ID = self.computeNode.node_name
        model_dir = os.path.join(self.computeNode.args.save, ID + 'save_models')
        model_path = os.path.join(model_dir, ID + 'checkpoint.pth.tar')
        if self.computeNode.epoch == 0:
            test_scores.append(self.score_header)
        val_loss, val_prec1 = validate(self.computeNode.val_loader, self.computeNode.model,
                                       self.computeNode.criterion, self.computeNode.args)
        if val_loss is None:
            print(val_loss)
        test_loss, test_prec1 = validate(full_test_loader, self.computeNode.model,
                                         self.computeNode.criterion, self.computeNode.args)

        etest_loss, etest_prec1, total_rate = validate_thinned(full_test_loader, self.computeNode.model,
                                                               self.computeNode.criterion,
                                                               self.computeNode.args,
                                                               self.computeNode.served_mask)

        # self.computeNode.move_model_to_cpu()
        # TODO: change this part to avoid over fitting. Check validation set accuracy.
        is_best = True

        train_stat = self.computeNode.train_stat
        today = datetime.now()
        score = f"{today.strftime('%d/%m/%y-%H:%M:%S')}\t{epoch}\t{self.computeNode.args.simulator_task}\t{self.computeNode.optimizer.param_groups[0]['lr']}\t"
        score += '\t'.join([f"{train_stat[e]['loss'].avg}" for e in range(self.E)])
        score += '\t'
        score += '\t'.join([f"{train_stat[e]['acc1'].avg}" for e in range(self.E)])
        score += '\t'
        score += '\t'.join([f"{val_loss[e].avg}" for e in range(self.E)])
        score += '\t'
        score += '\t'.join([f"{val_prec1[e].avg}" for e in range(self.E)])
        score += '\t'
        score += '\t'.join([f"{test_loss[e].avg}" for e in range(self.E)])
        score += '\t'
        score += '\t'.join([f"{test_prec1[e].avg}" for e in range(self.E)])
        score += f'\t{etest_loss}'
        score += f'\t{etest_prec1.sum / total_rate}'
        score += f'\t{total_rate}'
        score += f'\t{etest_prec1.count}'

        test_scores.append(score)
        scores_filename = os.path.join(self.computeNode.args.save, ID + 'scores.tsv')
        with open(scores_filename, 'a') as f:
            print('\n'.join(test_scores), file=f)
            f.flush()

        debug_filename = os.path.join(self.computeNode.args.save, ID + 'rates.tsv')
        with open(debug_filename, 'ab') as f:
            pickle.dump(self.computeNode.rates, f)
            f.flush()

        debug_filename = os.path.join(self.computeNode.args.save, ID + 'profiles.tsv')
        with open(debug_filename, 'ab') as f:
            pickle.dump(self.computeNode.profile, f)
            f.flush()

        logger.debug('Test:', test_scores)

        if self.computeNode.client_id == 0 and 'train' in self.computeNode.args.simulator_task:  # Only store cloud model and during training
            model_filename = 'checkpoint.pth.tar'
            save_checkpoint({
                'epoch': epoch,
                'arch': self.computeNode.args.arch,
                'state_dict': self.computeNode.model.state_dict(),
                'optimizer': self.computeNode.optimizer.state_dict(),
                'scheduler': self.computeNode.scheduler.state_dict(),
            }, self.computeNode.args, is_best, model_filename, self.computeNode.node_name)


def save_checkpoint(state, args, is_best, filename, node_name):
    ID = node_name  # + f'lr_{args.lr}'
    model_dir = os.path.join(args.save, ID + 'save_models')
    best_filename = os.path.join(model_dir, ID + 'best.txt')
    model_filename = os.path.join(model_dir, ID + filename)
    best_model_filename = os.path.join(model_dir, 'best_' + ID + filename)
    os.makedirs(args.save, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    logger.debug("=> saving checkpoint '{}'".format(model_filename))
    start = time.time()
    torch.save(state, model_filename)
    logger.debug(f'took: {time.time() - start:.2f}s')
    if is_best:
        # shutil.copyfile(model_filename, best_model_filename)
        with open(best_filename, 'wb') as fout:
            pickle.dump([state['epoch'], -1], fout)
    logger.debug("=> saved checkpoint '{}'".format(model_filename))
    return


def load_checkpoint(path, node_name):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ID = node_name
    model_dir = os.path.join(path, ID + 'save_models')
    filename = 'checkpoint.pth.tar'
    model_filename = os.path.join(model_dir,  ID + filename)



    print(model_filename)

    logger.debug("=> loading checkpoint '{}'".format(model_filename))
    state = None
    try:
        state = torch.load(model_filename, map_location=device)
        logger.debug("=> loaded checkpoint '{}'".format(model_filename))
    except:
        logger.error('=> failed to load checkpoint')
    print('State is None:', state is None)
    return state


def load_nodes_states(nodes, args):
    if args.resume:
        logger.debug('Resuming')
        epoch_file_name = os.path.join(args.save, 'successful_epoch.tsv')
        if os.path.exists(epoch_file_name):
            with open(epoch_file_name, 'r') as fin:
                epoch = int(fin.readlines()[0].strip())
        else:
            epoch = -1
            logger.debug('Cannot resume from a successful epoch')
        args.start_epoch = epoch + 1
        for node in nodes:
            logger.debug('loading state for node ', node.node_name)
            node.epoch = args.start_epoch
            checkpoint = load_checkpoint(args.save, node.node_name)
            if checkpoint is None and args.load_pretrained_model is not None:
                checkpoint = load_checkpoint(args.load_pretrained_model, node.node_name)
            if checkpoint is not None:
                node.model.load_state_dict(checkpoint['state_dict'])
                node.optimizer.load_state_dict(checkpoint['optimizer'])
                node.scheduler.load_state_dict(checkpoint['scheduler'])



def seal_epoch(epoch, args):
    result_filename = os.path.join(args.save, 'successful_epoch.tsv')
    with open(result_filename, 'w') as f:
        print(epoch, file=f)


class LambdaTildeStrategy(object):
    BIAS_ERR = 0
    EQUAL = 1
    FLOPS_PROP = 2
    OPT_ERR = 3
    GEN_ERR = 4
    ERR_ALL_PROD = 5
    ERR_ALL_MEAN = 6
    ERR_BIAS_OPT = 7
    ERR_BIAS_GEN = 8
    ERR_ALL_MED = 9
    ERR_BIAS_OPT_MEAN = 10
    ERR_MIN_BIAS_OPT_MEAN = 11
    ERR_EQ_ALL_MEAN = 12
    ERR_EQ_BIAS_OPT_MEAN = 13
    ERR_EQ_ALL_MED = 14
    OPT_ERR_MIN = 15
    OPT_ERR_EQ = 16


def calculate_flops(e, num_classes, layers=None):
    if not layers:
        layers = [3, 2, 2, 2]

    model = ResNetEE(
        layers_per_classifier=[e],
        num_classes=num_classes,
        layers=layers,
        nClassifiers=1,
        growth=2,
        channels=64,
    ).eval()
    model.set_nBlocks(1)
    input_tensor = torch.randn(1, 3, 32, 32)
    flops = FlopCountAnalysis(model, input_tensor)
    return flops.total()
