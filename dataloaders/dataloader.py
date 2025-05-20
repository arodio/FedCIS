import json
from typing import Any, Tuple

import numpy as np
import random
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import random_split, Subset
from PIL import Image
from copy import deepcopy

from torch.utils.data import ConcatDataset


def __getitem__(self, index: int) -> Tuple[Any, Any]:
    img, target = self.data[index], self.targets[index]

    # doing this so that it is consistent with all other datasets
    # to return a PIL Image
    img = Image.fromarray(np.array(img))

    if self.transform is not None:
        img = self.transform(img)

    if self.target_transform is not None:
        target = self.target_transform(target)

    return img, target


class DataLoader:
    def __init__(self, args):
        self.args = args
        with open(args.specs_dataset_split, 'r') as f:
            self.dataset_split = json.load(f)
        if args.data == 'cifar10':
            normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            augmented_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ])
            not_augmented_transform = transforms.Compose([
                transforms.ToTensor(),
                normalize
            ])

            if self.args.seed:
                torch.manual_seed(self.args.seed)
                np.random.seed(self.args.seed)

            if not args.disable_data_augmentation:
                trfs = augmented_transform
            else:
                trfs = not_augmented_transform

            train_set = datasets.CIFAR10(args.data_root, train=True,
                                         download=True,
                                         transform=trfs)
            test_set = datasets.CIFAR10(args.data_root, train=False,
                                        download=True,
                                        transform=transforms.Compose([
                                            transforms.ToTensor(),
                                            normalize
                                        ]))
        elif args.data == 'cifar100':
            normalize = transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))

            augmented_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ])

            not_augmented_transform = transforms.Compose([
                transforms.ToTensor(),
                normalize
            ])

            if self.args.seed:
                torch.manual_seed(self.args.seed)
                np.random.seed(self.args.seed)

            if not args.disable_data_augmentation:
                trfs = augmented_transform
            else:
                trfs = not_augmented_transform

            train_set = datasets.CIFAR100(args.data_root, train=True,
                                          download=True,
                                          transform=trfs)
            test_set = datasets.CIFAR100(args.data_root, train=False,
                                         download=True,
                                         transform=transforms.Compose([
                                             transforms.ToTensor(),
                                             normalize
                                         ]))
        elif args.data == 'mnist':
            datasets.MNIST.__getitem__ = __getitem__
            normalize = transforms.Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081))

            train_set = datasets.MNIST(args.data_root, train=True,
                                       download=True,
                                       transform=transforms.Compose([
                                           transforms.Resize((32, 32)),
                                           transforms.Grayscale(num_output_channels=3),  # Convert to 3 channel
                                           transforms.ToTensor(),
                                           normalize
                                       ]))
            test_set = datasets.MNIST(args.data_root, train=False,
                                      download=True,
                                      transform=transforms.Compose([
                                          transforms.Resize((32, 32)),
                                          transforms.Grayscale(num_output_channels=3),  # Convert to 3 channel
                                          transforms.ToTensor(),
                                          normalize
                                      ]))
        elif args.data == 'fashion-mnist':
            normalize = transforms.Normalize(
                (0.2860, 0.2860, 0.2860),(0.3530, 0.3530, 0.3530)
            )

            train_set = datasets.FashionMNIST(
                args.data_root, train=True, download=True,
                transform=transforms.Compose([
                   transforms.Resize((32, 32)),
                   transforms.Grayscale(num_output_channels=3),  # Convert to 3 channel
                   transforms.ToTensor(),
                   normalize
                ]))
            test_set = datasets.FashionMNIST(
                args.data_root, train=False, download=True,
                transform=transforms.Compose([
                  transforms.Resize((32, 32)),
                  transforms.Grayscale(num_output_channels=3),  # Convert to 3 channel
                  transforms.ToTensor(),
                  normalize
                ]))
        elif args.data == 'imagenette':
            normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

            augmented_transform = transforms.Compose([
                transforms.Resize((32, 32)),  # Resize to match CIFAR10 dimensions
                transforms.RandomCrop(32, padding=4),  # Apply random crop with padding
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ])

            not_augmented_transform = transforms.Compose([
                transforms.Resize((32, 32)),  # Resize to match CIFAR10 dimensions
                transforms.ToTensor(),
                normalize
            ])

            if self.args.seed:
                torch.manual_seed(self.args.seed)
                np.random.seed(self.args.seed)

            if not args.disable_data_augmentation:
                trfs = augmented_transform
            else:
                trfs = not_augmented_transform

            train_set = datasets.Imagenette(
                args.data_root, split='train', download=False, transform=trfs
            )
            test_set = datasets.Imagenette(
                args.data_root, split='val', download=False,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    normalize
                ]))
        else:
            raise Exception('Not implemented')

        if args.reduce_train_set_pct is not None:
            # Create a random subset with % of the data
            num_samples = len(train_set)
            num_reduced_samples = int(args.reduce_train_set_pct * num_samples)
            
            # Generate random indices for the subset
            indices = torch.randperm(num_samples)[:num_reduced_samples]

            # Reduce dataset
            train_set.data = train_set.data[indices]
            train_set.targets = [train_set.targets[i] for i in indices]

        train_set.targets = np.vstack((train_set.targets, np.arange(len(train_set.targets)))).T.astype(int)
        test_set.targets = np.vstack((test_set.targets, np.arange(len(test_set.targets)))).T.astype(int)
        self.train_set = train_set
        self.test_set = test_set

    def get_data(self, client_id):
        return self._get_dataloaders(client_id)

    def get_full_test_data(self):
        args = self.args
        test_set = self.test_set
        test_loader = torch.utils.data.DataLoader(
            test_set,
            sampler=torch.utils.data.sampler.SequentialSampler(list(range(test_set.data.shape[0]))),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
        return test_loader

    def _get_dataloaders(self, client_id):
        args = self.args
        train_set = self.train_set
        test_set = self.test_set
        # train_set = deepcopy(self.train_set)
        # test_set = deepcopy(self.test_set)

        # # Adjust batch_size to be proportional to dataset size - only matters for train_loader
        # if args.not_rescale_batch_size or client_id == 0:  # Set equal to the arg value
        #     batch_size = args.batch_size
        # else:  # Rescale the batch_size for all non-root clients (i.e., client_id != 0)
        #     batch_to_train_ratio = args.batch_size / len(self.dataset_split["node_dataset_assignment"][str(0)]['train'])
        #     batch_size = int(np.round(batch_to_train_ratio * len(self.dataset_split["node_dataset_assignment"][str(client_id)]['train'])))

        if self.args.single_batch_test:
            sampler_train = torch.utils.data.sampler.SequentialSampler(
                self.dataset_split["node_dataset_assignment"][str(client_id)]['train'])
        else:
            sampler_train = torch.utils.data.sampler.SubsetRandomSampler(
                self.dataset_split["node_dataset_assignment"][str(client_id)]['train'])

        if args.workers > 0 and not args.disable_data_augmentation:
            persistent_workers = True
        else:
            persistent_workers = False

        if not args.disable_data_augmentation:
            g_train = torch.Generator()
            g_train.manual_seed(self.args.seed)
            g_val = torch.Generator()
            g_val.manual_seed(self.args.seed)
            g_test = torch.Generator()
            g_test.manual_seed(self.args.seed)

            train_loader = torch.utils.data.DataLoader(
                train_set,
                batch_size=args.batch_size, shuffle=False, drop_last=True,
                sampler=sampler_train,
                num_workers=args.workers, pin_memory=False, persistent_workers=persistent_workers,
                worker_init_fn=seed_worker, generator=g_train,
            )
            sampler_val = torch.utils.data.sampler.SequentialSampler(
                self.dataset_split["node_dataset_assignment"][str(client_id)]['validation'])
            val_loader = torch.utils.data.DataLoader(
                train_set, batch_size=args.batch_size, shuffle=False,
                sampler=sampler_val,
                num_workers=args.workers, pin_memory=False, persistent_workers=persistent_workers,
                worker_init_fn=seed_worker, generator=g_val,
            )
            sampler_test = torch.utils.data.sampler.SequentialSampler(
                self.dataset_split["node_dataset_assignment"][str(client_id)]['test'])
            test_loader = torch.utils.data.DataLoader(
                test_set,
                sampler=sampler_test,
                batch_size=args.batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=False, persistent_workers=persistent_workers,
                worker_init_fn=seed_worker, generator=g_test,
            )
        else:
            train_loader = torch.utils.data.DataLoader(
                train_set,
                batch_size=args.batch_size, shuffle=False, drop_last=True,
                sampler=sampler_train,
                num_workers=0, pin_memory=False, persistent_workers=persistent_workers,
            )
            sampler_val = torch.utils.data.sampler.SequentialSampler(
                self.dataset_split["node_dataset_assignment"][str(client_id)]['validation'])
            val_loader = torch.utils.data.DataLoader(
                train_set, batch_size=args.batch_size, shuffle=False,
                sampler=sampler_val,
                num_workers=args.workers, pin_memory=False, persistent_workers=persistent_workers,
            )
            sampler_test = torch.utils.data.sampler.SequentialSampler(
                self.dataset_split["node_dataset_assignment"][str(client_id)]['test'])
            test_loader = torch.utils.data.DataLoader(
                test_set,
                sampler=sampler_test,
                batch_size=args.batch_size, shuffle=False,
                num_workers=0, pin_memory=False, persistent_workers=persistent_workers,
            )
            
        return train_loader, val_loader, test_loader


def renormalize(weights, index):
    """
    :param weights: vector of non-negative weights summing to 1.
    :type weights: numpy.array
    :param index: index of the weight to remove
    :type index: int
    """
    renormalized_weights = np.delete(weights, index)
    renormalized_weights /= renormalized_weights.sum()

    return renormalized_weights


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
