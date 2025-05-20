import argparse
import time

arg_parser = argparse.ArgumentParser(description='Federated Early Exits')

# ==================================== Experimental Configuration ====================================
exp_group = arg_parser.add_argument_group('exp', 'experiment setting')
exp_group.add_argument('--save', default='save/default-{}'.format(time.time()),
                       type=str, metavar='SAVE',
                       help='path to the experiment logging directory'
                            '(default: save/debug)')
exp_group.add_argument('--resume', action='store_true',
                       help='path to latest checkpoint (default: none)')
exp_group.add_argument('--load-pretrained-model', default=None, type=str, metavar='PATH',
                       help='path to saved checkpoint (default: none)')
exp_group.add_argument('--print-freq', '-p', default=50, type=int,
                       metavar='N', help='print frequency (default: 100)')
exp_group.add_argument('--stats-record-freq', '-sr', default=1, type=int,
                       metavar='N', help='print frequency (default: 100)')
exp_group.add_argument('--seed', default=0, type=int,
                       help='random seed')
exp_group.add_argument('--save_freq', default=1, type=int,
                       help='Recording stats frequency')
exp_group.add_argument('--gpu', default=0, type=str, help='# ID of target GPU')
exp_group.add_argument('--simulator-task', type=str, default='train',
                       choices=['train', 'train-single', 'evaluate', 'measure-exits'], help='Simulator task: train ('
                                                                                            'federated training), '
                                                                                            'train-single (single '
                                                                                            'client 2 classifiers training), '
                                                                                            'evaluate (compute stats '
                                                                                            'without training), '
                                                                                            'measure-exits (collect '
                                                                                            'FLOPs stats for current '
                                                                                            'e.e. configuration)')
exp_group.add_argument('--specs-topology', type=str, default='specs/configuration-exp.json',
                       help='Topology specifications (see, e.g., spec files)')
exp_group.add_argument('--specs-dataset-split', type=str, default='specs/configuration-test-cifar10-dataset.json',
                       help='Data split (train/val/test) spec file (see, e.g., spec files)')
exp_group.add_argument('--sampled-nodes', default=-1, type=int, metavar='D', help='Defaults to the number of nodes.')
exp_group.add_argument('--sampled-probability-per-layer', default='0.', type=str,
                       help='Assignment excludes either first or last exit, e.g., when sampled-reverse is False then 0.1-0.2 are probs for the second and third exits.')
exp_group.add_argument('--sampled-reverse', action='store_true',
                       help='Enable early exits to help later exits, instead of the default of later exits helping early exits.')
exp_group.add_argument('--single_client_p', default=0.5, type=float, help='Useful when train-single simulator'
                                                                          'task is selected. Externally select '
                                                                          'client serving weight on classifier 1')
exp_group.add_argument('--single_client_loss_rescale', action='store_true', help='Useful when train-single simulator '
                                                                                 'task is selected. It enables loss '
                                                                                 'rescaling by difficulty')
exp_group.add_argument('--multi_client_p', type=str, default=None,
                       help='Comma-separated list of client serving weights.')
exp_group.add_argument('--single_batch_test', action='store_true',
                       help='If set, train loader will use a single batch for testing purposes')
exp_group.add_argument('--lambda-tilde-strategy', default=0, type=int,
                       help='Assign the strategy for selecting lambda tilde. Must change corresponding enum in utils.')
exp_group.add_argument('--agg-strategy', default='pgrads', type=str,
                       help='Select strategy to use for aggregating client models.')
exp_group.add_argument('--not-rescale-batch-size', action='store_true',
                       help='If set, the batch size will NOT be rescaled.')
exp_group.add_argument('--save_experiment_data', action='store_true',
                       help='If set, will save data for the first several epochs.')
exp_group.add_argument('--float64', action='store_true', help='If set, use float64 as datatype, otherwise use float32.')
# ==================================== Data Configuration ====================================
data_group = arg_parser.add_argument_group('data', 'dataset setting')
data_group.add_argument('--data-root', metavar='DIR', default='data',
                        help='path to dataset (default: data)')
data_group.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')

data_group.add_argument('--data', default='cifar10',
                        choices=['cifar10', 'cifar100', 'mnist', 'imagenette', 'fashion-mnist'], help='Dataset')
data_group.add_argument('--reduce_train_set_pct', default=None, type=float)

# ==================================== Model Architecture Configuration ====================================

arch_group = arg_parser.add_argument_group('arch', 'Resnet model architecture setting')
arch_group.add_argument('--arch', default='resnet', type=str, help='Select model')
arch_group.add_argument('--resnet-layers-widths', default='2-2-2-2', type=str,
                        help='Customize resnet layers per block')  # Customize resnet layers per block
arch_group.add_argument('--resnet-nChannels', type=int, default=64,
                        help=' Customize resnet nChannels')  # Customize resnet nChannels
arch_group.add_argument('--resnet-growthRate', type=int, default=2,
                        help="Customize resnet channels growth rate ")  # Customize resnet channels growth rate
arch_group.add_argument('--resnet-layers_per_classifier', default='3-5-7',
                        type=str, help="Layers to classifier assigment. Layers assigned to a classifier should be "
                                       "strictly increasing (e.g., 2-4-5, 2 layers assigned to classifier 1, "
                                       "4 layers assigned to classifier 2, 5 layers assigned to classifier 3)")
arch_group.add_argument('--no_batch_norm', action='store_true',
                        help='If set, the model will have BatchNorm layers removed.')

# ==================================== Optimiser Configuration ====================================
optim_group = arg_parser.add_argument_group('optimization',
                                            'optimization setting')
optim_group.add_argument('--epochs', default=50, type=int, metavar='N',
                         help='number of total epochs to run (default: 50)')
optim_group.add_argument('--lr-milestones', default='20-40', type=str)
optim_group.add_argument('--start-epoch', default=0, type=int, metavar='N',
                         help='manual epoch number (useful on restarts)')
optim_group.add_argument('-b', '--batch-size', default=128, type=int,
                         metavar='N', help='mini-batch size (default: 128)')
optim_group.add_argument('--optimizer', default='sgd', choices=['sgd', 'rmsprop', 'adam'], metavar='N',
                         help='optimizer (default=sgd)')
optim_group.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR',
                         help='initial learning rate (default: 0.1)')
optim_group.add_argument('--lr-type', default='multistep', type=str, metavar='T',
                         help='learning rate strategy (default: multistep)',
                         choices=['cosine', 'multistep', 'fixed'])
# optim_group.add_argument('--decay-rate', default=0.5, type=float, metavar='N',
#                          help='decay rate of learning rate (default: 0.1)')
optim_group.add_argument('--momentum', default=0.9, type=float, metavar='M',
                         help='momentum (default=0.9)')
optim_group.add_argument('--weight-decay', '--wd', default=0.0, type=float, metavar='W',
                         help='weight decay (default: 0 / 5e-4)')
optim_group.add_argument('--loss-rescaling', type=str, default='loss',
                         choices=['loss', 'accuracy', 'confusion'])
optim_group.add_argument('--local-epochs', default=1, type=int, metavar='E', help='number of local epochs')

optim_group.add_argument('--disable-data-augmentation', action='store_true', help='Disable data augmentation. It '
                                                                                  'might provide inconsistent '
                                                                                  'difficulty perception when enabled '
                                                                                  '(same point may be considered '
                                                                                  'difficult depending on the '
                                                                                  'augmentation operation)')

# ==================================== Early-Exit Configuration ====================================
ee_group = arg_parser.add_argument_group('ee', 'early exit configuration')
ee_group.add_argument('--confidence-type', default='entropy', type=str,
                      help='Confidence Type: entropy / max')  # Customize resnet layers per block
# ==================================== Rec ====================================
