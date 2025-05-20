import pickle
import time

import numpy as np
import torch
from loguru import logger

from models.resnets import ResNetEE
from trainingtopologies.custom_training_topology import Train
from utils.opt_counter import measure_model
from utils.utils import load_args


def main():
    args = load_args()  # Setup args

    logger.debug(f'Used Cuda Device: {torch.cuda.current_device()}, {torch.cuda.device_count()}')
    clients_tree = Train(args)  # Load client topology
    clients_tree.load_nodes_states()  # Load previous states
    logger.debug(f'Model sizes: {clients_tree.model_sizes}')
    logger.debug(f'Exits count: {clients_tree.exit_count}')
    # clients_tree.broadcast(args.start_epoch)  # Broadcast model from root node
    if args.simulator_task == 'train':  # Train federated early exit network
        for epoch in range(args.start_epoch, args.epochs):
            logger.debug('=' * 50)
            logger.debug(f'Epoch: {epoch}')
            logger.debug(f'Sample Clients')
            clients_tree.sample()  # samples clients (Algo. 1, line 3)
            logger.debug(f'Exits count: {clients_tree.exit_count}')
            logger.debug(f'Sampling Schedule: {clients_tree.sampled_nodes_exits}')
            logger.debug(f'Refresh profiles:')
            clients_tree.compute_profiles()
            clients_tree.compute_loss_rescale()
            logger.debug(f'loss rescale alpha {clients_tree.nodes[0].loss_rescale_alphas}')
            logger.debug(f'Local Update:')
            t = time.perf_counter()
            for n in clients_tree.nodes:  # Train each node independently (Algo. 1, lines 7-9)
                if n in clients_tree.sampled_nodes_exits:
                    e = clients_tree.sampled_nodes_exits[n]
                    if e in clients_tree.rates.lambda_e:
                        # Training strategies rescale the number of local epochs proportional to nodes' dataset size
                        local_epochs_n = round(args.local_epochs * (
                                    clients_tree.nodes[0].dataset_size / clients_tree.nodes[n].dataset_size))
                        clients_tree.nodes[n].local_update(e, outer_epoch=epoch,
                                                           local_epochs=local_epochs_n)  # Perform local update on exit e at client n
                else:
                    clients_tree.nodes[n].scheduler.step()  # decrease the learning rate even when not sampled
            logger.info(time.perf_counter() - t)
            logger.debug(f'Aggregate:')
            clients_tree.aggregate(epoch)  # Aggregate local updates (Algo. 1, lines 11-12)
            logger.debug(f'Broadcast:')
            clients_tree.broadcast(epoch)  # Broadcast root model (Algo. 1, line 4)
            if epoch % args.save_freq == 0 or epoch == args.epochs - 1:
                clients_tree.record_stats(epoch)  # Record accuracy statistics
    elif args.simulator_task == 'train-single':  # Train federated early exit network
        for epoch in range(args.start_epoch, args.epochs):
            logger.debug('=' * 50)
            logger.debug(f'Epoch: {epoch}')
            logger.debug(f'Sample Clients')
            clients_tree.sample()  # samples clients
            logger.debug(f'Sampling Schedule: {clients_tree.sampled_nodes_exits}')
            logger.debug(f'Refresh profiles:')
            logger.debug(f'Local Update:')
            t = time.perf_counter()
            scale = clients_tree.nodes[0].eval_accurate_rescale(args.single_client_p)  # Evaluate the loss rescaling
            if args.multi_client_p:
                scale_serving_weights = np.array(
                    [float(p) for p in args.multi_client_p.split(',')])  # Scale with multi-exit serving weights only
            else:
                scale_serving_weights = np.array(
                    [args.single_client_p, 1 - args.single_client_p])  # Scale only with by serving weights

            # new lines of code to extend train-single to multi-exit case
            if not args.single_client_loss_rescale:
                scale = scale_serving_weights
            else:
                scale *= scale_serving_weights  # Further scale by serving weights

            scale /= sum(scale)  # Normalize rescaling weights
            logger.debug(f'Scale: {str(scale)}')
            clients_tree.nodes[0].local_update_single_client(scale,
                                                             outer_epoch=epoch)  # Perform a local update with provided scale
            logger.info(time.perf_counter() - t)
            clients_tree.broadcast(epoch)  # Broadcast model (this is required only for stat. recording)
            if epoch % args.save_freq == 0 or epoch == args.epochs - 1:
                clients_tree.record_stats(epoch)  # Record stat
    elif args.simulator_task == 'evaluate':  # Evaluate Accuracies
        clients_tree.record_stats(np.nan)  # Don't train just refresh stats
    elif args.simulator_task == 'measure-exits':  # Code to collect # of flops per classifier
        model = ResNetEE(args.layers_widths, nClassifiers=args.nClassifiers, growth=args.growthRate,
                         channels=args.nChannels, num_classes=args.num_classes)
        input = torch.ones((1, 3, 32, 32))  # Dummy input
        cls_ops, cls_params = measure_model(model, 32, 32)  # Measure model ops and params sizes
        with open(args.save + '/model-cifar10.stats', 'wb') as file:
            pickle.dump((cls_ops, cls_params), file)


if __name__ == '__main__':
    main()
