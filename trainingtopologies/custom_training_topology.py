import copy
import json
import os
import pickle
import types

import networkx as nx
import numpy as np
import pandas as pd
import torch
from loguru import logger
from collections import defaultdict

from dataloaders.dataloader import DataLoader
from fedclients.aggregator import Aggregator
from utils.utils import load_args, NodeStats, seal_epoch, load_checkpoint, LambdaTildeStrategy, calculate_flops

np.set_printoptions(precision=3)


def served_proportion(lambda_h, fraction):
    lambda_s = np.zeros(lambda_h.size)
    mass = np.sum(lambda_h) * fraction
    for _i in range(lambda_h.size):
        if lambda_h[_i] >= mass:
            lambda_s[_i] = mass
            return lambda_s
        else:
            mass -= lambda_h[_i]
            lambda_s[_i] = lambda_h[_i]
    return lambda_s


def get_exit(block, layers_per_classifier):
    # res = np.where(np.array(list(range(3, 17, 2))) == block + 1)[0]
    res = np.where(layers_per_classifier == block + 1)[0]
    if len(res) == 0:
        return -1
    else:
        return res[0]


class Train:
    def __init__(self, args):
        self.args = args
        self.single_node = False
        with open(args.specs_topology, 'r') as f:
            specs = json.load(f)
            graph = nx.DiGraph()
            edges = [tuple(e) for e in specs['toplogy_as_edges']]
            graph.add_edges_from(edges)
            graph.add_node(0)
            self.n_clients = len(graph.nodes)
            self.nodes = {}
            self.model_sizes = {}
            data_loader = DataLoader(args)
            args.nClassifiers = specs['node_model_size'][str(0)]
            if args.sampled_nodes == -1:
                args.sampled_nodes = self.n_clients
            self.full_test_loader = data_loader.get_full_test_data()

            for n in range(len(graph.nodes)):
                nEE = specs['node_model_size'][str(n)]
                self.nodes[n] = Aggregator(nEE=nEE,
                                           loader=data_loader,
                                           epoch=args.start_epoch,
                                           args=args,
                                           children=list(graph.successors(n)),
                                           parent=list(graph.predecessors(n)),
                                           client_id=n)

                # # check the upcoming loader batches
                # print(f"Node: {n} - Train")
                # cur_train_loader = iter(self.nodes[n].train_loader)
                # for i in range(5):
                #     x, y = next(cur_train_loader)
                #     print(x[0][1][10][:10])
                #     print(y[:5, 0])

                self.model_sizes[n] = nEE  # at this point the data loaders are the exact same
                self.nodes[n].difficulty_levels = specs['difficulty_quantization']
                self.nodes[n].profile = np.zeros(self.nodes[n].difficulty_levels)
                self.nodes[n].rates = types.SimpleNamespace()
                self.nodes[n].rates.arrival = specs['rates']['arrival'][str(n)]
                self.nodes[n].rates.departure = specs['rates']['departure'][str(n)]
                self.nodes[n].serving_weights_rescale = specs['serving_weights_rescale']
                self.nodes[n].loss_rescale = specs['loss_rescale']
                self.nodes[n].rescale_by_flow_map = specs['rescale_by_flow_map']
                logger.debug('Loss rescale: ', self.nodes[n].loss_rescale)
                logger.debug('Dataset size:', self.nodes[n].dataset_size)

        # Set global model
        self.cloud_model = copy.deepcopy(self.nodes[0].model).to('cuda' if torch.cuda.is_available() else 'cpu')
        self.relative_flops_es = None
        self.gen_error_es = None
        self.opt_error_es = None
        self.rescale_by_flow_map = specs['rescale_by_flow_map']
        self.loss_rescale = specs['loss_rescale']
        self.nodes_stats = [NodeStats(n) for n in self.nodes.values()]
        self.topology = graph
        logger.debug(" Compute fractions:")
        for n in range(len(self.nodes)):
            node = self.nodes[n]
            if node.rates.departure == 0:
                node.rates.fraction = 1
            else:
                node.rates.fraction = np.max(
                    [0, 1 - node.rates.departure / (node.rates.arrival + sum(
                        [self.nodes[_n].rates.departure for _n in list(self.topology.successors(n))]))])
        self.rates = types.SimpleNamespace()
        self.rates.s_e = {}
        self.rates.lambda_e = {}
        # Associate nodes to their corresponding level in the training tree.
        self.associate_nodes_level = {0: [0]}

        def _recursive_associate(n, level, associate_nodes_level):
            succ = list(self.topology.successors(n))
            if len(succ) != 0:
                for _n in succ:
                    if level + 1 in associate_nodes_level:
                        associate_nodes_level[level + 1].append(_n)
                    else:
                        associate_nodes_level[level + 1] = [_n]
                    _recursive_associate(_n, level + 1, associate_nodes_level)

        _recursive_associate(0, 0, self.associate_nodes_level)
        for e in range(1, 1 + self.args.nClassifiers):
            nodes_e = list(filter(lambda n: self.nodes[n].nEE == e, self.nodes))
            self.rates.s_e[e - 1] = 0
            for _n in nodes_e:
                self.rates.s_e[e - 1] += np.max([0, self.nodes[_n].rates.arrival - self.nodes[_n].rates.departure + sum(
                    [self.nodes[__n].rates.departure for __n in self.topology.successors(_n)])])

        self.specs = specs

    def compute_loss_rescale(self):
        if self.loss_rescale:
            alphas = np.zeros((self.nodes[0].nEE, self.nodes[0].difficulty_levels))
            for n in self.nodes:
                e, node = self.nodes[n].nEE - 1, self.nodes[n]
                print(sum(node.rates.lambda_es[e]), node.node_name, e)
                alpha = node.rescale_weight(node.rates.lambda_es[e] / sum(node.rates.lambda_es[e]), e)
                alphas[e] += alpha
            self.loss_rescale_alphas = {e: np.sum(alphas[e]) for e in range(self.nodes[0].nEE)}
            _normalizer = sum([self.loss_rescale_alphas[e] for e in range(self.nodes[0].nEE)])
            self.loss_rescale_alphas = {e: self.loss_rescale_alphas[e] / _normalizer for e in range(self.nodes[0].nEE)}
        else:
            self.loss_rescale_alphas = {e: 0 for e in range(self.nodes[0].nEE)}

        for n in self.nodes:
            self.nodes[n].loss_rescale_alphas = self.loss_rescale_alphas

    def compute_profiles(self):
        leaves_to_cloud = np.flip(range(len(self.associate_nodes_level)))
        for level in leaves_to_cloud:
            nodes = self.associate_nodes_level[level]
            for n in nodes:
                node = self.nodes[n]

                if node.difficulty_levels == 1:
                    node.profile = np.ones(1)
                else:
                    node.eval_difficulty_profile()
                # node.eval_difficulty_profile()

                profile = node.profile
                logger.debug(node.node_name, node.rates)
                lambda_h = node.rates.arrival * profile
                if len(list(self.topology.successors(n))) != 0:
                    # Flow strategy
                    if self.rescale_by_flow_map:
                        for _n in self.topology.successors(n):
                            if self.nodes[_n].nEE == node.nEE:
                                lambda_h += self.nodes[_n].lambda_t
                            else:
                                mass_i = np.copy(self.nodes[_n].lambda_t)
                                for e in range(self.nodes[_n].nEE - 1, node.nEE - 1):
                                    T = node.T[(e, e + 1)]
                                    mass_j = np.zeros(node.difficulty_levels)
                                    for (i, j) in T:
                                        mass_j[j] += T[(i, j)] * mass_i[i]

                                    mass_i = mass_j
                            lambda_h += mass_j
                    else:
                        # lambda_h is all points received
                        # add to this value all the points transmitted from a node's children
                        lambda_h += np.sum(np.vstack([self.nodes[_n].lambda_t for _n in self.topology.successors(n)]),
                                           axis=0)
                lambda_s = served_proportion(lambda_h, node.rates.fraction)
                # lambda_h = what we generate locally + what child forwards
                # lambda_s = what we serve at our code
                # lambda_t = what a node overall transmits
                lambda_t = lambda_h - lambda_s
                node.lambda_s = lambda_s
                node.lambda_t = lambda_t

        # Cloud part
        # lambda_e is total rate served by given exit
        self.rates.lambda_e = {}  # np.zeros(self.args.nClassifiers)
        for e in range(1, 1 + self.args.nClassifiers):
            nodes_e = list(filter(lambda n: self.nodes[n].nEE == e, self.nodes))
            if len(nodes_e) != 0:
                self.rates.lambda_e[e - 1] = np.sum(np.vstack([(self.nodes[n].lambda_s) for n in nodes_e]), axis=0)
        print('SE', self.rates.s_e)
        for n in self.nodes:
            self.nodes[n].rates.lambda_es = self.rates.lambda_e
            self.nodes[n].rates.s_es = self.rates.s_e

    def sample(self):
        sampled_nodes = np.sort(np.random.choice(list(self.nodes), size=self.args.sampled_nodes, replace=False))
        self.sampled_nodes_exits = {}
        for n in sampled_nodes:
            node = self.nodes[n]
            p = node.sampled_probability
            if np.random.choice([0, 1], p=[1 - p, p]) == 0:
                self.sampled_nodes_exits[n] = node.nEE - 1
                self.exit_count[node.nEE - 1] += 1
            else:
                if self.args.sampled_reverse:
                    self.sampled_nodes_exits[n] = np.random.choice(list(range(node.nEE, self.nodes[0].nEE)))
                    self.exit_count[self.sampled_nodes_exits[n]] += 1
                else:
                    self.sampled_nodes_exits[n] = np.random.choice(list(range(node.nEE - 1))) if node.nEE > 1 else 0
                    self.exit_count[self.sampled_nodes_exits[n]] += 1

        debug_filename = os.path.join(self.nodes[0].args.save, 'sampling_scheme.tsv')
        with open(debug_filename, 'ab') as f:
            pickle.dump(self.sampled_nodes_exits, f)
            f.flush()

    def broadcast(self, epoch):
        with torch.no_grad():
            for n in self.nodes:
                self.nodes[n].epoch = epoch
                model = self.nodes[n].model
                for name in model.state_dict():
                    self.nodes[n].model.state_dict()[name].copy_(self.cloud_model.state_dict()[name].data.clone())

    def update_aggregate(self):
        for n in self.nodes:
            self.nodes[n].update_whole()
            self.nodes[n].epoch += 1
            if isinstance(self.nodes[n], Aggregator):
                self.nodes[n].aggregate()

    def record_stats(self, epoch):
        leaves_to_cloud = np.flip(range(len(self.associate_nodes_level)))
        for level in leaves_to_cloud:
            nodes = self.associate_nodes_level[level]
            for n in nodes:
                self.nodes[n].serveable = np.zeros(len(self.full_test_loader.dataset))
                self.nodes[n].serveable[self.nodes[n].test_loader.sampler.data_source] = self.nodes[
                                                                                             n].rates.arrival / len(
                    self.nodes[n].test_loader.sampler.data_source)
                for _n in self.topology.successors(n):
                    self.nodes[n].serveable += self.nodes[_n].serveable
                total_serving_rate = self.nodes[n].rates.arrival + np.sum(
                    [self.nodes[_n].rates.departure for _n in self.topology.successors(n)]) - self.nodes[
                                         n].rates.departure
                total_serving_rate = max([0, total_serving_rate])
                self.nodes[n].determine_serving_schedule(self.full_test_loader, total_serving_rate)

        result_filename = os.path.join(self.args.save, 'exit_count.pk')
        with open(result_filename, 'wb') as f:
            pickle.dump(self.exit_count, f)
            f.flush()

        for stats in self.nodes_stats:
            logger.debug(stats.computeNode.node_name)
            stats.save_stats(epoch, self.full_test_loader)
            stats.computeNode.reset_train_stats()
        if not np.isnan(epoch):
            seal_epoch(epoch, self.args)

    def load_nodes_states(self):

        epoch_file_name = os.path.join(self.args.save, 'successful_epoch.tsv')
        if os.path.exists(epoch_file_name):
            with open(epoch_file_name, 'r') as fin:
                epoch = int(fin.readlines()[0].strip())
            result_filename = os.path.join(self.args.save, 'exit_count.pk')
            with open(result_filename, 'rb') as f:
                self.exit_count = pickle.load(f)
        else:
            self.exit_count = np.zeros(self.nodes[0].nEE)
            epoch = -1
            logger.debug('No previous experiement exits')
        for node in self.nodes.values():
            ID = node.node_name
            classifier_weights = os.path.join(node.args.save, ID + 'classifier_stats.tsv')
            if os.path.exists(classifier_weights):
                logger.debug(pd.read_csv(classifier_weights, sep='\t').values[-1, 1:])
                node.classifiers_weight = pd.read_csv(classifier_weights, sep='\t').values[-1, 1:]
        for node in self.nodes.values():
            ID = node.node_name  # + f'lr_{args.lr}'
            model_dir = os.path.join(node.args.save, ID + 'save_models')
            best_filename = os.path.join(model_dir, ID + 'best.txt')
            if os.path.exists(best_filename):
                with open(best_filename, 'rb') as fin:
                    _, best_acc = pickle.load(fin)
                # logger.debug(best_acc, 'test')
        self.args.start_epoch = epoch + 1
        for node in self.nodes.values():
            logger.debug('loading state for node ', node.node_name)
            node.epoch = self.args.start_epoch
            logger.debug('model moving to gpu')
            node.move_model_to_gpu()
            logger.debug('model moved to gpu')
            checkpoint = load_checkpoint(self.args.save, node.node_name)
            if checkpoint is None and self.args.load_pretrained_model is not None:  # Experiment is intiated with a a pretrained model in a different path
                state = load_checkpoint(self.args.load_pretrained_model, node.node_name)['state_dict']

                for name in state:
                    node.model.state_dict()[name].copy_(state[name].data)
                # node.model.load_state_dict(
                # )
                logger.debug('=> loaded checkpoint')
                # raise  Exception('Test Exception')
            if checkpoint is not None:  # Load previous checkpoint
                logger.debug('Load optimizer  and scheduler')
                # node.model.load_state_dict(checkpoint['state_dict'])
                # state = checkpoint['state_dict']
                # print(state['blocks.0.0.weight'].flatten()[:10])
                # for name in state:
                    # node.model.state_dict()[name].copy_(state[name].data)
                node.model.load_state_dict(checkpoint['state_dict'])
                node.optimizer.load_state_dict(checkpoint['optimizer'])
                node.scheduler.load_state_dict(checkpoint['scheduler'])
            if not isinstance(node, Aggregator):
                node.move_model_to_cpu()
        self.cloud_model = self.nodes[0].model
    def aggregate(self, epoch):
        """
        Aggregation rule (lines 11-12 of Algorithm 1)
        """
        if self.args.agg_strategy == 'pgrads':
            logger.debug(f'cloud model block weights (sample): {self.cloud_model.blocks[7].conv1.weight[5][1][2]}')
            logger.debug(f'cloud model classifier weights (sample): {self.cloud_model.classifiers[9].fc.weight[2][14:17]}')

            if not self.gen_error_es and self.args.lambda_tilde_strategy in [
                LambdaTildeStrategy.GEN_ERR, LambdaTildeStrategy.ERR_BIAS_GEN, LambdaTildeStrategy.ERR_ALL_PROD,
                LambdaTildeStrategy.ERR_ALL_MEAN, LambdaTildeStrategy.ERR_ALL_MED,
                LambdaTildeStrategy.ERR_EQ_ALL_MEAN, LambdaTildeStrategy.ERR_EQ_ALL_MED
            ]:
                total_flops_es = {}
                for e, e_layer in enumerate(self.nodes[0].model.layers_per_classifier):
                    total_flops_es[e] = calculate_flops(e_layer, self.args.num_classes)
                dataset_size_es = {e: 0 for e in range(self.nodes[0].nEE)}
                for k, v in self.sampled_nodes_exits.items():
                    dataset_size_es[v] += self.nodes[k].dataset_size
                self.gen_error_es = {i: np.sqrt(dataset_size_es[i] / total_flops_es[i]) for i in range(self.nodes[0].nEE)}
                self.gen_error_es = {k: v / sum(self.gen_error_es.values()) for k, v in self.gen_error_es.items()}
                logger.debug(f'gen_error_es: {self.gen_error_es}')
            if not self.opt_error_es and self.args.lambda_tilde_strategy in [
                LambdaTildeStrategy.OPT_ERR, LambdaTildeStrategy.ERR_BIAS_OPT, LambdaTildeStrategy.ERR_ALL_PROD,
                LambdaTildeStrategy.ERR_ALL_MEAN, LambdaTildeStrategy.ERR_ALL_MED, LambdaTildeStrategy.ERR_BIAS_OPT_MEAN,
                LambdaTildeStrategy.ERR_MIN_BIAS_OPT_MEAN, LambdaTildeStrategy.ERR_EQ_ALL_MEAN,
                LambdaTildeStrategy.ERR_EQ_BIAS_OPT_MEAN, LambdaTildeStrategy.ERR_EQ_ALL_MED,
                LambdaTildeStrategy.OPT_ERR_MIN, LambdaTildeStrategy.OPT_ERR_EQ
            ]:
                if self.args.arch == 'resnet':
                    if list(self.nodes[0].model.layers_per_classifier) == [3, 9, 17]:
                        if self.args.lambda_tilde_strategy in [
                            LambdaTildeStrategy.ERR_MIN_BIAS_OPT_MEAN, LambdaTildeStrategy.OPT_ERR_MIN
                        ]:
                            self.opt_error_es = {
                                0: 1 / (0.00374**2 * 0.25**2 * 4),
                                1: 1 / (0.00224**2 * 0.5**2 * 2),
                                2: 1 / 0.00101**2
                            }
                        elif self.args.lambda_tilde_strategy in [
                            LambdaTildeStrategy.ERR_EQ_ALL_MEAN, LambdaTildeStrategy.ERR_EQ_BIAS_OPT_MEAN,
                            LambdaTildeStrategy.ERR_EQ_ALL_MED, LambdaTildeStrategy.OPT_ERR_EQ
                        ]:
                            self.opt_error_es = {
                                0: 1 / (0.00374 * np.sqrt(0.25**2 * 4)),
                                1: 1 / (0.00224 * np.sqrt(0.5**2 * 2)),
                                2: 1 / 0.00101
                            }
                    elif list(self.nodes[0].model.layers_per_classifier) == [9, 13, 17]:
                        if self.args.lambda_tilde_strategy in [
                            LambdaTildeStrategy.ERR_MIN_BIAS_OPT_MEAN, LambdaTildeStrategy.OPT_ERR_MIN
                        ]:
                            self.opt_error_es = {
                                0: 1 / (0.00216**2 * 0.25**2 * 4),
                                1: 1 / (0.00126**2 * 0.5**2 * 2),
                                2: 1 / 0.00101**2
                            }
                        elif self.args.lambda_tilde_strategy in [
                            LambdaTildeStrategy.ERR_EQ_ALL_MEAN, LambdaTildeStrategy.ERR_EQ_BIAS_OPT_MEAN,
                            LambdaTildeStrategy.ERR_EQ_ALL_MED, LambdaTildeStrategy.OPT_ERR_EQ
                        ]:
                            self.opt_error_es = {
                                0: 1 / (0.00216 * np.sqrt(0.25**2 * 4)),
                                1: 1 / (0.00126 * np.sqrt(0.5**2 * 2)),
                                2: 1 / 0.00101
                            }
                    else:
                        raise NotImplementedError("Must calculate gradient variances for new model config.")
                    self.opt_error_es = {k: v / sum(self.opt_error_es.values()) for k, v in self.opt_error_es.items()}
                    logger.debug(f'opt_error_es: {self.opt_error_es}')
                else:
                    raise NotImplementedError("Must calculate gradient variances for new model config.")

            exits_t = np.sort(np.unique(list(self.sampled_nodes_exits.values())))
            lambda_es = self.nodes[0].rates.s_es
            if self.args.lambda_tilde_strategy == LambdaTildeStrategy.BIAS_ERR:
                lambda_tilde_es = copy.deepcopy(lambda_es)
            elif self.args.lambda_tilde_strategy == LambdaTildeStrategy.EQUAL:
                lambda_tilde_es = {e: 1 / self.nodes[0].nEE for e in range(self.nodes[0].nEE)}
            elif self.args.lambda_tilde_strategy == LambdaTildeStrategy.FLOPS_PROP:
                if not self.relative_flops_es:
                    total_flops_es = {}
                    for e, e_layer in enumerate(self.nodes[0].model.layers_per_classifier):
                        total_flops_es[e] = calculate_flops(e_layer, self.args.num_classes)
                    self.relative_flops_es = {k: v / total_flops_es[self.nodes[0].nEE - 1] for k, v in total_flops_es.items()}
                tau_i = 0.01 * (epoch + 1)
                lambda_tilde_es = {}
                for e in range(self.nodes[0].nEE):
                    if tau_i > self.relative_flops_es[e]:
                        lambda_tilde_es[e] = self.relative_flops_es[e]
                    else:
                        lambda_tilde_es[e] = tau_i
            elif self.args.lambda_tilde_strategy in [
                LambdaTildeStrategy.OPT_ERR, LambdaTildeStrategy.OPT_ERR_MIN, LambdaTildeStrategy.OPT_ERR_EQ
            ]:
                lambda_tilde_es = {e: self.opt_error_es[e] for e in range(self.nodes[0].nEE)}
            elif self.args.lambda_tilde_strategy == LambdaTildeStrategy.GEN_ERR:
                lambda_tilde_es = {e: self.gen_error_es[e] for e in range(self.nodes[0].nEE)}
            elif self.args.lambda_tilde_strategy in [
                LambdaTildeStrategy.ERR_ALL_PROD, LambdaTildeStrategy.ERR_ALL_MEAN, LambdaTildeStrategy.ERR_ALL_MED,
                LambdaTildeStrategy.ERR_EQ_ALL_MEAN, LambdaTildeStrategy.ERR_EQ_ALL_MED
            ]:
                lambda_es = {k: v / sum(lambda_es.values()) for k, v in lambda_es.items()}
                lambda_tilde_es = {}
                if self.args.lambda_tilde_strategy == LambdaTildeStrategy.ERR_ALL_PROD:
                    lambda_tilde_es = {e: lambda_es[e] * self.gen_error_es[e] * self.opt_error_es[e] for e in range(self.nodes[0].nEE)}
                elif self.args.lambda_tilde_strategy in [
                    LambdaTildeStrategy.ERR_ALL_MEAN, LambdaTildeStrategy.ERR_EQ_ALL_MEAN
                ]:
                    lambda_tilde_es = {e: (lambda_es[e] + self.gen_error_es[e] + self.opt_error_es[e]) / 3 for e in range(self.nodes[0].nEE)}
                elif self.args.lambda_tilde_strategy in [
                    LambdaTildeStrategy.ERR_ALL_MED, LambdaTildeStrategy.ERR_EQ_ALL_MED
                ]:
                    lambda_tilde_es = {e: np.median([lambda_es[e], self.gen_error_es[e], self.opt_error_es[e]]) for e in range(self.nodes[0].nEE)}
            elif self.args.lambda_tilde_strategy == LambdaTildeStrategy.ERR_BIAS_OPT:
                lambda_es = {k: v / sum(lambda_es.values()) for k, v in lambda_es.items()}
                lambda_tilde_es = {e: lambda_es[e] * self.opt_error_es[e] for e in range(self.nodes[0].nEE)}
            elif self.args.lambda_tilde_strategy in [
                LambdaTildeStrategy.ERR_BIAS_OPT_MEAN, LambdaTildeStrategy.ERR_MIN_BIAS_OPT_MEAN,
                LambdaTildeStrategy.ERR_EQ_BIAS_OPT_MEAN
            ]:
                lambda_es = {k: v / sum(lambda_es.values()) for k, v in lambda_es.items()}
                lambda_tilde_es = {e: (lambda_es[e] + self.opt_error_es[e]) / 2 for e in range(self.nodes[0].nEE)}
            else:
                raise NotImplementedError("Must add new strategy to LambdaTildeStrategy. ")

            # Normalize lambda_es to sum to 1
            lambda_tilde_es = {k: v / sum(lambda_tilde_es.values()) for k, v in lambda_tilde_es.items()}
            logger.debug(f'lambda_tilde_es: {lambda_tilde_es}')
            eta_s = 1

            with torch.no_grad():
                w_diff = copy.deepcopy(self.cloud_model)
                zero_state_dict = {key: torch.zeros_like(value) for key, value in w_diff.state_dict().items()}
                w_diff.load_state_dict(zero_state_dict)

                # Calculate the differences btw the global model's weights and weighted client models' weights
                for e in exits_t:
                    clients_t_e = [k for k, v in self.sampled_nodes_exits.items() if v == e]
                    # clients_t_e_ds_size = {c: self.nodes[c].dataset_size for c in clients_t_e}
                    e_block = self.cloud_model.layers_per_classifier[e]
                    lambda_tilde_e = lambda_tilde_es[e]
                    for c in clients_t_e:
                        e_ds_size = np.sum([node.dataset_size for node in self.nodes.values() if e == (node.nEE - 1) or (e < (node.nEE - 1) and node.sampled_probability > 0.)])
                        c_ds_scale_factor = self.nodes[c].dataset_size / e_ds_size
                        if self.nodes[c].nEE == 1:
                            p_ce = 1.
                        elif e == (self.nodes[c].nEE - 1):
                            p_ce = 1 - self.nodes[c].sampled_probability
                        else:
                            p_ce = self.nodes[c].sampled_probability / (self.nodes[c].nEE - 1)
                        for block in range(e_block):
                            for name in w_diff.blocks[block].state_dict():
                                if 'num_batches_tracked' in name:
                                    continue
                                w_diff.blocks[block].state_dict()[name].add_(
                                    lambda_tilde_e * c_ds_scale_factor * p_ce *
                                    (self.nodes[c].model.blocks[block].state_dict()[name].data.clone() -
                                     self.cloud_model.blocks[block].state_dict()[name].data.clone())
                                )
                        # Update classifier for last block in range
                        for name in w_diff.classifiers[block].state_dict():
                            w_diff.classifiers[block].state_dict()[name].add_(
                                lambda_tilde_e * c_ds_scale_factor * p_ce *
                                (self.nodes[c].model.classifiers[block].state_dict()[name].data.clone() -
                                 self.cloud_model.classifiers[block].state_dict()[name].data.clone())
                            )

                # Update the global model using the calculated differences
                for block in range(self.cloud_model.nBlocks):
                    for name in self.cloud_model.blocks[block].state_dict():
                        if 'num_batches_tracked' in name:
                            # This should be okay because we ensure that all clients have same number of batches
                            self.cloud_model.blocks[block].state_dict()[name].copy_(
                                self.nodes[0].model.blocks[block].state_dict()[name])
                        else:
                            self.cloud_model.blocks[block].state_dict()[name].add_(
                                eta_s * w_diff.blocks[block].state_dict()[name].data.clone()
                            )
                    if block in self.cloud_model.layers_per_classifier:
                        for name in self.cloud_model.classifiers[block].state_dict():
                            self.cloud_model.classifiers[block].state_dict()[name].add_(
                                eta_s * w_diff.classifiers[block].state_dict()[name].data.clone()
                            )
                # logger.debug(f'cloud model block weights (sample): {self.cloud_model.blocks[7].conv1.weight[5][1][2]}')
                # logger.debug(f'cloud model classifier weights (sample): {self.cloud_model.classifiers[9].fc.weight[2][14:17]}')

            # No longer needs to be set because we use self.cloud_model
            # self.nodes[0].model = copy.deepcopy(self.cloud_model)

        else:
            raise ValueError(f"Unrecognized Aggregation Strategy: {self.args.agg_strategy}")




if __name__ == '__main__':
    args = load_args()
    t = Train(args)
    pass
