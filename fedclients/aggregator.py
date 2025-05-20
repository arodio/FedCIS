import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
from alive_progress import alive_bar
from loguru import logger
import pickle
from copy import deepcopy

from fedclients.computeNode import ComputeNode, AverageMeter, accuracy


class Aggregator(ComputeNode):
    def __init__(self, nEE, loader, epoch, children, parent, args, client_id=0):
        super(Aggregator, self).__init__(nEE=nEE, loader=loader, epoch=epoch, args=args, client_id=client_id)
        # self.specifications_clients_dict = specifications_clients_dict
        self.children = (children)
        if len(parent) == 0:  # root
            self.parent = -1
        else:
            self.parent = parent[0]  # Tree topologies

        # for c_list in specifications_clients_dict.values():
        #     self.clients.extend(c_list)
        self.active = True
        self.reset_train_stats()

        # only for experiment to validate CL vs FL
        self.infinite_train_loader_initialized = False
        self.infinite_train_loader = None
        if self.args.simulator_task == 'train-single':
            self.losses_list = {"3": [], "17": []}
            self.grads_list = {"3": [], "17": [], "combined": []}
        elif self.args.simulator_task == 'train':
            self.losses_list = []
            self.grads_list = []
        self.weights_list = []
        self.local_weights_list = []
        self.data_points = []

    # only for experiment to validate CL vs FL
    def infinite_loader(self, loader):
        while True:
            for batch in loader:
                yield batch

    def local_update_single_client(self, scale, outer_epoch=None):
        logger.debug(f'Local update {self.node_name}', self.args.loss_rescaling)
        self.move_model_to_gpu()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = [AverageMeter() for _ in range(self.model.nClassifiers)]
        top1, top3 = [], []
        for i in range(self.model.nClassifiers):
            top1.append(AverageMeter())
            top3.append(AverageMeter())
        ee_top1, ee_top3 = [], []
        for i in range(self.model.nClassifiers):
            ee_top1.append(AverageMeter())
            ee_top3.append(AverageMeter())
        self.model.train()
        end = time.time()
        with alive_bar(self.args.local_epochs * len(self.train_loader), ctrl_c=False, title=f'Local training',
                       force_tty=True) as bar:
            for l_i in range(self.args.local_epochs):

                train_iterator = iter(self.train_loader)
                for k_i in range(len(train_iterator)):

                    if self.args.single_batch_test and self.args.disable_data_augmentation is False:
                        torch.manual_seed(len(self.train_loader) * outer_epoch + k_i)
                        np.random.seed(len(self.train_loader) * outer_epoch + k_i)

                    input, target_index = next(train_iterator)

                    data_time.update(time.time() - end)

                    input = input.to(self.device)
                    if self.args.float64:
                        input = input.double()
                    target = target_index[:, 0].type(torch.long).to(self.device)

                    # input_var = torch.autograd.Variable(input)
                    # target_var = torch.autograd.Variable(target)
                    input_var = input
                    target_var = target

                    self.optimizer.zero_grad()

                    output = self.model(input_var)
                    if not isinstance(output, list):
                        output = [output]

                    loss = 0.

                    for e in range(self.nEE):
                        L = self.criterion(output[e], target_var)
                        # if outer_epoch == 0 and k_i % 10 == 0 and k_i < 100:
                        if self.args.save_experiment_data and outer_epoch == 0 and k_i < 20:
                            if e == 0:
                                self.losses_list["3"].append(L.item())
                                L.backward(retain_graph=True)
                                self.grads_list["3"].append(
                                    {name: param.grad.clone() for name, param in self.model.named_parameters() if param.grad is not None})
                                self.optimizer.zero_grad()
                            elif e == 1:
                                self.data_points.append(input)
                                self.losses_list["17"].append(L.item())
                                L.backward(retain_graph=True)
                                self.grads_list["17"].append(
                                    {name: param.grad.clone() for name, param in self.model.named_parameters() if param.grad is not None})
                                self.optimizer.zero_grad()
                        _loss = L * scale[e]
                        loss += _loss
                        losses[e].update(_loss.item(), input.size(0))

                    loss.backward()
                    # Previously updating .grad manually based on recordings
                    # if self.args.save_experiment_data and outer_epoch == 0 and k_i < 20:
                    #     for name, param in self.model.named_parameters():
                    #         if name.split('.')[0] == 'classifiers' and int(name.split('.')[1]) not in [2, 16]:
                    #             pass
                    #         elif name in self.grads_list["3"][k_i].keys():
                    #             if name.split('.')[0] == 'classifiers':
                    #                 param.grad = 0.9 * self.grads_list["3"][k_i][name]
                    #             else:
                    #                 param.grad = 0.9 * self.grads_list["3"][k_i][name] + 0.1 * self.grads_list["17"][k_i][name]
                    #         else:
                    #             param.grad = 0.1 * self.grads_list["17"][k_i][name]

                    for j in range(len(output)):
                        prec1, prec3 = accuracy(output[j].data, target, topk=(1, 3))
                        top1[j].update(prec1.item(), input.size(0))
                        top3[j].update(prec3.item(), input.size(0))

                    # if outer_epoch == 0 and k_i % 10 == 0 and k_i < 100:
                    if self.args.save_experiment_data and outer_epoch == 0 and k_i < 20:
                        self.weights_list.append({name: param.data.clone() for name, param in self.model.named_parameters()})

                    # if outer_epoch == 0 and k_i == 99:
                    if self.args.save_experiment_data and outer_epoch == 0 and k_i == 19:
                        print("Finished recording")
                        self.losses_list = {k: np.array(v) for k, v in self.losses_list.items()}
                        torch.save(self.losses_list, 'losses_cl.pt')
                        torch.save(self.data_points, 'data_points_cl.pt')
                        torch.save(self.grads_list, 'grads_cl.pt')
                        torch.save(self.weights_list, 'weights_cl.pt')

                    self.optimizer.step()
                    bar()
                for e in range(self.nEE):
                    self.train_stat[e] = {'loss': losses[e], 'acc1': top1[e]}
                    logger.info(f'{self.node_name}' + '.Epoch[{epoch}]:\t'
                                                      'Loss {loss.val:.2f}\t'
                                                      'Acc@1 {top1.val:.2f}\t'
                                                      'Acc@3 {top3.val:.2f}'.format(
                        epoch=self.epoch, loss=losses[e], top1=top1[e], top3=top3[e]))
            self.scheduler.step()

    def local_update(self, e, outer_epoch=None, local_epochs=None):
        logger.debug(f'Local update {self.node_name}', self.args.loss_rescaling)
        self.move_model_to_gpu()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = [AverageMeter() for _ in range(self.model.nClassifiers)]
        top1, top3 = [], []
        for i in range(self.model.nClassifiers):
            top1.append(AverageMeter())
            top3.append(AverageMeter())
        ee_top1, ee_top3 = [], []
        for i in range(self.model.nClassifiers):
            ee_top1.append(AverageMeter())
            ee_top3.append(AverageMeter())
        self.model.train()
        end = time.time()
        with alive_bar(local_epochs * len(self.train_loader), ctrl_c=False, title=f'Local training',
                       force_tty=True) as bar:
            for l_i in range(local_epochs):

                if self.args.single_batch_test:
                    if not self.infinite_train_loader_initialized:
                        # self.infinite_train_loader = self.infinite_loader(deepcopy(self.train_loader))
                        self.infinite_train_loader = self.infinite_loader(self.train_loader)
                        self.infinite_train_loader_initialized = True
                    train_loader = self.infinite_train_loader
                else:
                    train_loader = self.train_loader

                if self.args.single_batch_test and self.args.disable_data_augmentation is False:
                    torch.manual_seed(outer_epoch)
                    np.random.seed(outer_epoch)

                for k_i, (input, target_index) in enumerate(train_loader):

                    # if outer_epoch % 10 == 0 and outer_epoch < 100:
                    if self.args.save_experiment_data and outer_epoch < 20:
                        if self.node_name == "Node_0_":
                            self.data_points.append(input)

                    data_time.update(time.time() - end)
                    input = input.to(self.device)
                    if self.args.float64:
                        input = input.double()
                    target = target_index[:, 0].type(torch.long).to(self.device)

                    input_var = input
                    target_var = target
                    self.optimizer.zero_grad()

                    output = self.model(input_var)
                    if not isinstance(output, list):
                        output = [output]

                    if self.loss_rescale:
                        alpha = self.loss_rescale_alphas[e]
                    else:
                        alpha = 1  # / len(self.loss_rescale_alphas)
                    L = self.criterion(output[e], target_var)

                    loss = L * alpha  # / L.detach() * alpha
                    losses[e].update(loss.item(), input.size(0))

                    for j in range(len(output)):
                        prec1, prec3 = accuracy(output[j].data, target, topk=(1, 3))
                        top1[j].update(prec1.item(), input.size(0))
                        top3[j].update(prec3.item(), input.size(0))
                    # compute gradient and do SGD step
                    loss.backward()

                    # if outer_epoch % 10 == 0 and outer_epoch < 100:
                    if self.args.save_experiment_data and outer_epoch < 20:
                        self.losses_list.append(loss.item())
                        self.weights_list.append(
                            {name: param.data.clone() for name, param in self.model.named_parameters()}
                        )

                        self.grads_list.append(
                            {name: param.grad.clone() for name, param in self.model.named_parameters() if
                             param.grad is not None}
                        )

                    self.optimizer.step()

                    if self.args.save_experiment_data and outer_epoch < 20:
                        self.local_weights_list.append(
                            {name: param.data.clone() for name, param in self.model.named_parameters()}
                        )

                    # if outer_epoch == 99:
                    if self.args.save_experiment_data and outer_epoch == 19:
                        print("Finished recording losses")
                        if self.node_name == "Node_0_":
                            save_name = "fl_17"
                        elif self.node_name == "Node_1_":
                            save_name = "fl_3"
                        torch.save(self.losses_list, f'losses_{save_name}.pt')
                        torch.save(self.grads_list, f'grads_{save_name}.pt')
                        torch.save(self.weights_list, f'weights_{save_name}.pt')
                        torch.save(self.local_weights_list, f'local_weights_{save_name}.pt')
                        if self.node_name == "Node_0_":
                            torch.save(self.data_points, 'data_points_fl.pt')

                    bar()

                    if self.args.single_batch_test:
                        break

                if e == -1:
                    for _e in range(self.nEE):
                        self.train_stat[_e] = {
                            'loss': losses[_e],
                            'acc1': top1[_e]
                        }
                else:
                    self.train_stat[e] = {
                        'loss': losses[e],
                        'acc1': top1[e]
                    }

            if not self.args.single_batch_test:
                self.scheduler.step()
        logger.info(f'{self.node_name}' + '.Epoch[{epoch}]:\t'
                                          'Loss {loss.val:.2f}\t'
                                          'Acc@1 {top1.val:.2f}\t'
                                          'Acc@3 {top3.val:.2f}'.format(
            epoch=self.epoch, loss=losses[e], top1=top1[e], top3=top3[e]))

    def broadcast(self):
        for client in self.children:
            # if client.active:
            client.classifiers_weight = self.classifiers_weight[: len(client.classifiers_weight)]
            for name, params in zip((client.model).state_dict(), (client.model).parameters()):
                params.data = self.model.state_dict()[name].data.clone()

    def eval_accurate_rescale(self, p_served):
        self.pts_confs = {}
        self.pts_indices = {}
        self.pts_losses = {}

        for e in range(self.nEE):
            self.pts_confs[e] = []
            self.pts_indices[e] = []
            self.pts_losses[e] = []

        with torch.no_grad():
            self.model.eval()
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            for i, (input, target) in enumerate(self.train_loader):
                target = target.to(device)
                input = input.to(device)
                input_var = torch.autograd.Variable(input)
                target_var = torch.autograd.Variable(target[:, 0])
                output = self.model(input_var)
                if not isinstance(output, list):
                    output = [output]
                if not isinstance(output, list):
                    output = [output]
                for e in range(self.nEE):
                    ee_output = F.softmax(output[e], dim=1).detach()
                    confidence = torch.sum(-ee_output * torch.log(ee_output), dim=1) / torch.log(
                        torch.tensor(float(self.args.num_classes)).to(self.device))
                    loss = F.cross_entropy(output[e], target_var, reduction='none').detach()
                    indices = target[:, 1]
                    self.pts_confs[e].extend(confidence.cpu().numpy())
                    self.pts_indices[e].extend(indices.cpu().numpy())
                    self.pts_losses[e].extend(loss.cpu().numpy())

        for e in range(self.nEE):
            self.pts_confs[e] = np.array(self.pts_confs[e])
            self.pts_indices[e] = np.array(self.pts_indices[e])
            self.pts_losses[e] = np.array(self.pts_losses[e])

        _confs_ids = np.argsort(self.pts_confs[0])
        pts = self.pts_indices[0][_confs_ids][: int(p_served * self.pts_indices[0].size)]
        mask_0 = np.isin(self.pts_indices[0], pts)
        mask_1 = ~np.isin(self.pts_indices[1], pts)

        scale = np.array([np.mean(self.pts_losses[0][mask_0]) / np.mean(self.pts_losses[0]),
                              np.mean(self.pts_losses[1][mask_1]) / np.mean(self.pts_losses[1])])
        self.pts_to_exit = np.zeros(np.max(self.pts_indices[0]) + 1)
        self.pts_to_exit[self.pts_indices[0][mask_0]] = 0
        self.pts_to_exit[self.pts_indices[0][mask_1]] = 1
        return scale

    def eval_difficulty_profile(self):
        self.pts_confs = {}
        self.pts_indices = {}
        for e in range(self.nEE):
            self.pts_confs[e] = []
            self.pts_indices[e] = []

        with torch.no_grad():
            self.model.eval()
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            for i, (input, target) in enumerate(self.train_loader):
                target = target.to(device)
                input = input.to(device)
                input_var = torch.autograd.Variable(input)
                target_var = torch.autograd.Variable(target[:, 0])
                output = self.model(input_var)
                if not isinstance(output, list):
                    output = [output]
                if not isinstance(output, list):
                    output = [output]
                for e in range(self.nEE):
                    ee_output = F.softmax(output[e], dim=1).detach()
                    confidence = torch.sum(-ee_output * torch.log(ee_output), dim=1) / torch.log(
                        torch.tensor(float(self.args.num_classes)).to(self.device))
                    indices = target[:, 1]
                    self.pts_confs[e].extend(confidence.cpu().numpy())
                    self.pts_indices[e].extend(indices.cpu().numpy())
        self.T = {}
        self.pts_diff = {}
        for e in range(self.nEE):
            self.pts_confs[e] = np.array(self.pts_confs[e])
            self.pts_indices[e] = np.array(self.pts_indices[e])
            self.diff_bins = np.arange(0,
                                       1 + 1 / self.difficulty_levels,
                                       1 / self.difficulty_levels)
            _pts_diff = np.digitize(self.pts_confs[e], self.diff_bins) - 1
            pts_diff = np.ones(len(self.train_loader.dataset)) * -1
            pts_diff[self.pts_indices[e]] = _pts_diff
            self.pts_diff[e] = pts_diff
            if e > 0:
                self.T[(e - 1, e)] = {}
                i_s = list(filter(lambda i: sum(self.pts_diff[e - 1] == i) != 0, range(self.difficulty_levels)))
                j_s = list(filter(lambda i: sum(self.pts_diff[e] == i) != 0, range(self.difficulty_levels)))
                for i in i_s:
                    for j in j_s:
                        self.T[(e - 1, e)][i, j] = (len(set(np.where(self.pts_diff[e - 1] == i)[0]).intersection(
                            set(np.where(self.pts_diff[e] == j)[0])))) / sum(self.pts_diff[e - 1] == i)
        
        if self.rescale_by_flow_map:  # Flow Based
            self.diff_bins = np.arange(0,
                                       1 + 1 / self.difficulty_levels,
                                       1 / self.difficulty_levels)
            self.profile, _ = np.histogram(self.pts_confs[self.nEE - 1], bins=self.diff_bins)
            logger.debug('DEBUG:', len(self.profile), _)
            self.profile = self.profile / np.sum(self.profile)

        else:  # Percentage based
            self.profile = np.ones(self.difficulty_levels) / self.difficulty_levels
            indices = self.pts_indices[self.nEE - 1][np.argsort(self.pts_confs[self.nEE - 1])]
            # self.pts_diff = -np.ones(len(self.train_loader.dataset))
            # self.pts_diff[indices] = ((np.arange(indices.size) / indices.size) * self.difficulty_levels).astype(int)
	
	    # proposed fix to keep self.pts_diff a dictionary instead of overwriting it as an array
            self.pts_diff[e] = -np.ones(len(self.train_loader.dataset))
            self.pts_diff[e][indices] = ((np.arange(indices.size) / indices.size) * self.difficulty_levels).astype(int)

    def rescale_weight(self, lambda_e, e):
        loss = np.zeros(self.difficulty_levels)
        loss_N = np.zeros(self.difficulty_levels)
        # Split dataset to difficulty profile.
        with torch.no_grad():
            self.model.eval()
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            for i, (input, target) in enumerate(self.train_loader):
                target = target.to(device)
                input = input.to(device)
                input_var = torch.autograd.Variable(input)
                target_var = torch.autograd.Variable(target[:, 0])
                output = self.model(input_var)
                if not isinstance(output, list):
                    output = [output]
                if not isinstance(output, list):
                    output = [output]
                l = F.cross_entropy(output[e], target_var, reduction='none').detach()
                batch_indexes = target[:, 1].cpu().detach()

		        # error because self.pts_diff is an array when rescale_by_flowmap is False - resolved
                batch_diffs = self.pts_diff[e][batch_indexes]

                for k in range(self.difficulty_levels):
                    if l[k == batch_diffs].size(0) != 0:
                        loss[k] += torch.sum(l[k == batch_diffs]).cpu()
                        loss_N[k] += l[k == batch_diffs].size(0)
        return np.array(
            [(lambda_e[k] * loss[k]) / loss_N[k] if loss_N[k] != 0 else 0 for k in
             range(self.difficulty_levels)]) / (sum(loss) / sum(loss_N))

    def determine_serving_schedule(self, test_loader, total_serving_rate):  # Test
        pts_confs = []
        pts_indices = []
        with torch.no_grad():
            self.model.eval()
            for i, (input, target) in enumerate(test_loader):
                target = target.to(self.device)
                input = input.to(self.device)
                input_var = torch.autograd.Variable(input)
                target_var = torch.autograd.Variable(target[:, 0])
                output = self.model(input_var)
                if not isinstance(output, list):
                    output = [output]
                if not isinstance(output, list):
                    output = [output]
                ee_output = F.softmax(output[-1], dim=1).detach()
                if self.args.confidence_type =='entropy':
                    confidence = torch.sum(-ee_output * torch.log(ee_output), dim=1) / torch.log(
                        torch.tensor(float(self.args.num_classes)).to(self.device))
                else:
                    confidence = -torch.max(ee_output, dim=1)[0] # Min: -1 and Max: 0
                indices = target[:, 1]
                pts_confs.extend(confidence.cpu().numpy())
                pts_indices.extend(indices.cpu().numpy())
        pts_confs = np.array(pts_confs)
        pts_indices = np.array(pts_indices)
        sorted_indexes = np.argsort(pts_confs)  # easy to difficult
        sorted_indexes = pts_indices[sorted_indexes]
        _mask = self.serveable.astype(bool)[sorted_indexes]
        _considered_pts = sorted_indexes[_mask]
        res = np.where(np.cumsum(self.serveable[_considered_pts]) >= total_serving_rate - 1e-10)[0]
        if len(res) != 0:
            limit_index = res[0]
        else:
            limit_index = len(_considered_pts) - 1
        remainder = np.max([np.sum(self.serveable[_considered_pts][:limit_index + 1]) - total_serving_rate, 0])

        logger.debug(self.node_name)
        self.serveable[_considered_pts[:limit_index + 1]] -= remainder / len(_considered_pts[:limit_index + 1])
        self.serveable[_considered_pts[limit_index:]] += remainder / len(_considered_pts[limit_index:])
        served_indexes = _considered_pts[:limit_index + 1]
        self.served_mask = np.zeros(self.serveable.size)
        self.served_mask[served_indexes] = self.serveable[served_indexes]

        self.serveable[served_indexes] = 0
        # self.serveable[_considered_pts][limit_index] = np.max([remainder, 0])
        logger.debug(
            f'Served: {sum(self.served_mask)} pts, Total Rate: {total_serving_rate}')
        # Evaluate Acc

    def reset_train_stats(self):
        self.train_stat = {}
        for e in range(self.model.nClassifiers):
            placeholder = SimpleNamespace()
            placeholder.avg = '-'
            self.train_stat[e] = {'loss': placeholder, 'acc1': placeholder}
