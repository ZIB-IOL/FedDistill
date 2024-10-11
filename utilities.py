# ===========================================================================
# Project:      On the Byzantine-Resilience of Distillation-Based Federated Learning - IOL Lab @ ZIB
# Paper:        arxiv.org/abs/2402.12265
# File:         utilities.py
# Description:  Utility functions
# ===========================================================================
from __future__ import print_function

import json
import os
import os.path
import sys
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from PIL import Image
from torchmetrics.classification import MulticlassAccuracy as Accuracy

from bisect import bisect_right

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle


class Utilities:
    """Class of utility functions"""

    @staticmethod
    def fill_dict_with_none(d):
        for key in d:
            if isinstance(d[key], dict):
                Utilities.fill_dict_with_none(d[key])  # Recursive call for nested dictionaries
            else:
                d[key] = None
        return d

    @staticmethod
    def update_config_with_default(configDict, defaultDict):
        """Update config with default values recursively."""
        for key, default_value in defaultDict.items():
            if key not in configDict:
                configDict[key] = default_value
            elif isinstance(default_value, dict):
                configDict[key] = Utilities.update_config_with_default(configDict.get(key, {}), default_value)
        return configDict

    @staticmethod
    def dump_bar_plot_to_wandb(x: list, y: list, xlabel: str, ylabel: str, title: str, wandb_identifier: str):
        """Dump a bar plot to wandb."""
        plt.bar(x, y)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        wandb.log({wandb_identifier: wandb.Image(plt)}, commit=False)
        plt.clf()

    @staticmethod
    def dump_dict_to_json_wandb(dumpDict, name):
        """Dump some dict to json and upload it"""
        fPath = os.path.join(wandb.run.dir, f'{name}.json')
        with open(fPath, 'w') as fp:
            json.dump(dumpDict, fp)
        wandb.save(fPath)

    @staticmethod
    def calculate_communication_cost(tensorList: list[torch.Tensor]) -> int:
        """Returns the communication cost of a list of tensors in bytes."""
        total_bytes = 0
        for x in tensorList:
            bytes_per_number = x.element_size()
            total_bytes += x.numel() * bytes_per_number
        return int(total_bytes)

    @staticmethod
    def get_model_communication_cost(model: torch.nn.Module) -> int:
        """Returns the communication cost of a model in bytes."""
        return Utilities.calculate_communication_cost(list(model.parameters()))

    @staticmethod
    def get_index_dataset(OriginalDataset: torch.utils.data.dataset.Dataset):
        """
        Returns Overloaded Dataset class that returns index of current image as well.
        If unlabeled is True, don't return label.
        Args:
            OriginalDataset (torch.utils.data.dataset.Dataset): Dataset to overload

        Returns: Overloaded dataset class
        """

        class AlteredDatasetWrapper(OriginalDataset):

            def __init__(self, *args, **kwargs):
                super(AlteredDatasetWrapper, self).__init__(*args, **kwargs)

            def __getitem__(self, index):
                # Overload this to collect the class indices once in a vector, which can then be used in the sampler
                item = super(AlteredDatasetWrapper, self).__getitem__(index=index)

                # If the dataset is unlabeled, we just return None as the label
                if isinstance(item, tuple):
                    # Labels exist
                    image, label = item
                else:
                    image = item
                    label = None
                return image, label, index

        AlteredDatasetWrapper.__name__ = OriginalDataset.__name__
        return AlteredDatasetWrapper

    @staticmethod
    def reset_dataset_subset_indices(dataset: torch.utils.data.dataset.Subset):
        """Resets indices of dataset, especially Subsets such that [2534, 19, 125] can become [0, 1, 2]"""
        dataset.indices = list(range(len(dataset.indices)))

    @staticmethod
    def get_preinitialized_dataset(OriginalDataset: torch.utils.data.Dataset, **settings) -> torch.utils.data.Dataset:
        """
        Overloads the OriginalDataset to preset some values at initialization, given by settings dictionary
        Args:
            OriginalDataset (torch.utils.data.Dataset): Dataset to overload
            **settings: Arbitrary number of settings in key-val-format

        Returns: Overloaded dataset class
        """

        class Preinitialized_Dataset(OriginalDataset):
            def __init__(self, **kwargs):
                super().__init__(**settings, **kwargs)

        Preinitialized_Dataset.__name__ = OriginalDataset.__name__
        return Preinitialized_Dataset

    @staticmethod
    def get_client_models(clients: list) -> list[OrderedDict]:
        return [client.model.state_dict() for client in clients]

    @staticmethod
    @torch.no_grad()
    def average_client_models(client_model_list: list) -> OrderedDict:
        """
        Average the weights of all client models.
        Returns: averaged state_dict
        """
        average_state_dict = OrderedDict()
        factor = 1.0 / len(client_model_list)
        for client_state_dict in client_model_list:
            for key in client_state_dict:
                if key not in average_state_dict:
                    average_state_dict[key] = (factor * client_state_dict[key].clone().detach())
                else:
                    average_state_dict[key] += (factor * client_state_dict[key].clone().detach())

        return average_state_dict

    @staticmethod
    @torch.no_grad()
    def average_client_predictions(client_predictions_list: list, output_type: str) -> torch.Tensor:
        """
        Average the predictions of all clients. output_type can control:
        the averaged probabilities (over all clients, 'softmax'), the corresponding prediction ('soft_prediction')
                or the most frequent prediction ('hard_prediction').
        Returns: averaged tensor
        """
        compute_avg_probs = (output_type in ['softmax', 'soft_prediction'])
        store_tensor = torch.zeros_like(client_predictions_list[0],
                                        device=client_predictions_list[0].device)  # On CUDA for now

        for client_predictions in client_predictions_list:
            if compute_avg_probs:
                # Just add the probabilities for the average
                store_tensor += client_predictions
            else:
                store_tensor += torch.nn.functional.one_hot(torch.argmax(client_predictions, dim=1),
                                                            num_classes=client_predictions.shape[1]).float()

        if output_type == 'softmax':
            # Weighted average of probabilities
            store_tensor.mul_(1. / len(client_predictions_list))  # Weighting
        elif output_type in ['soft_prediction', 'hard_prediction']:
            # Take the prediction given average probabilities (no need to actually average)
            store_tensor = torch.argmax(store_tensor, dim=1)

            # Convert to one-hot
            store_tensor = torch.nn.functional.one_hot(store_tensor,
                                                       num_classes=client_predictions_list[0].shape[1]).float()

        return store_tensor

    @staticmethod
    @torch.no_grad()
    def filter_outlier_scores(pred_stack: torch.Tensor):
        """
        Compute the outlier scores along the maximum variance eigenvector of the Covariance matrix
        Returns: tensor with outlier scores, dim: [dataset_size, n_clients, 1]
        """
        pred_stack = pred_stack.permute(1, 2, 0)
        batch_cov = torch.func.vmap(torch.cov)
        cov = batch_cov(pred_stack)
        ev = Utilities.top_ev(cov)
        mean_pred = torch.mean(pred_stack, dim=2, keepdim=True)
        centered_mean = (pred_stack - mean_pred).permute(0, 2, 1)  # reshape(ds_size,n_clients,n_classes)
        return torch.matmul(centered_mean, ev)

    @staticmethod
    @torch.no_grad()
    def filtered_mean(pred_stack: torch.Tensor, outlier_scores: torch.Tensor, threshold: torch.Tensor):
        """
        Compute the filtered mean given the outlier scores and the threshold(s)
        Returns: tensor with filtered mean, dim: [dataset_size, n_classes]
        """
        pred_stack = pred_stack.permute(1, 2, 0)
        mask = 1 * (torch.abs(outlier_scores.squeeze()) <= threshold)
        pred_masked = pred_stack * mask.unsqueeze(1)
        n_nonfiltered = torch.sum(mask, dim=1)
        filtered_mean = torch.sum(pred_masked, dim=2) / n_nonfiltered.unsqueeze(-1)
        idx_all_filtered = torch.nonzero((torch.sum(mask, dim=1)) == 0)
        # use the mean for samples where all predictions are over the outlier score
        if idx_all_filtered.nelement() > 0:
            mean_pred = torch.mean(pred_stack, dim=2, keepdim=True)
            filtered_mean[idx_all_filtered, :] = mean_pred[idx_all_filtered, :, 0]
        return filtered_mean

    @staticmethod
    @torch.no_grad()
    def top_ev(K, n_power_iterations=400, dim=1, eps=1e-10):
        v = torch.ones(K.shape[0], K.shape[1], 1).to(K.device)
        for _ in range(n_power_iterations):
            m = torch.bmm(K, v)
            n = torch.norm(m, dim=1).unsqueeze(1)
            v = m / torch.clamp(n, min=eps)
        return v

    @staticmethod
    @torch.no_grad()
    def geomedian(pred_stack, max_iter=100, eps=1e-12, ftol=1e-8):
        pred_stack = pred_stack.permute(1, 2, 0)  # -> ds_size, n_classes, n_clients
        # init gm at mean
        gm = torch.mean(pred_stack, dim=2, keepdim=True)

        for i in range(max_iter):
            dist_to_median = pred_stack - gm
            norms = torch.linalg.norm(dist_to_median, dim=1)
            if i > 0: obj_val_last = obj_val_new
            obj_val_new = torch.sum(norms, dim=1)
            inv_norms = 1. / torch.clamp(norms, min=eps)
            sum_of_inv_norms = torch.sum(inv_norms, dim=1)
            fractions = pred_stack / torch.clamp(norms.unsqueeze(1), min=eps)
            sum_of_fractions = torch.sum(fractions, dim=2)
            gm = (sum_of_fractions / sum_of_inv_norms.unsqueeze(1)).unsqueeze(2)

            # check for tolerance
            if i > 0:
                if torch.all(torch.gt(ftol * obj_val_new, torch.abs(obj_val_last - obj_val_new))):
                    break
        return gm.squeeze(2), norms

    @staticmethod
    @torch.no_grad()
    def get_highest_byz(n_total):
        assert n_total > 0, 'n_total has to be a positive integer'
        half = n_total/2
        half_mod = n_total%2
        if half_mod == 0:
            return int(half-1)
        else:
            return int(half)


class ImageNetDownSample(torch.utils.data.Dataset):
    """`DownsampleImageNet`_ Dataset. Taken from https://github.com/ma-xu/SparseSENet/blob/master/imagenetLoad.py
    """

    train_list = [
        ['train_data_batch_1'],
        ['train_data_batch_2'],
        ['train_data_batch_3'],
        ['train_data_batch_4'],
        ['train_data_batch_5'],
        ['train_data_batch_6'],
        ['train_data_batch_7'],
        ['train_data_batch_8'],
        ['train_data_batch_9'],
        ['train_data_batch_10']
    ]
    test_list = [
        ['val_data'],
    ]

    def __init__(self, root, train=True,
                 transform=None, target_transform=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        # now load the picked numpy arrays
        if self.train:
            self.train_data = []
            self.train_labels = []
            for fentry in self.train_list:
                f = fentry[0]
                file = os.path.join(self.root, f)
                fo = open(file, 'rb')
                if sys.version_info[0] == 2:
                    entry = pickle.load(fo)
                else:
                    entry = pickle.load(fo, encoding='latin1')
                self.train_data.append(entry['data'])
                if 'labels' in entry:
                    self.train_labels += entry['labels']
                else:
                    self.train_labels += entry['fine_labels']
                fo.close()
            # resize label range from [1,1000] to [0,1000),
            # This is required by CrossEntropyLoss
            self.train_labels[:] = [x - 1 for x in self.train_labels]

            self.train_data = np.concatenate(self.train_data)
            [picnum, pixel] = self.train_data.shape
            pixel = int(np.sqrt(pixel / 3))
            self.train_data = self.train_data.reshape((picnum, 3, pixel, pixel))
            self.train_data = self.train_data.transpose((0, 2, 3, 1))  # convert to HWC
        else:
            f = self.test_list[0][0]
            file = os.path.join(self.root, f)
            fo = open(file, 'rb')
            if sys.version_info[0] == 2:
                entry = pickle.load(fo)
            else:
                entry = pickle.load(fo, encoding='latin1')
            self.test_data = entry['data']
            [picnum, pixel] = self.test_data.shape
            pixel = int(np.sqrt(pixel / 3))

            if 'labels' in entry:
                self.test_labels = entry['labels']
            else:
                self.test_labels = entry['fine_labels']
            fo.close()

            # resize label range from [1,1000] to [0,1000),
            # This is required by CrossEntropyLoss
            self.test_labels[:] = [x - 1 for x in self.test_labels]
            self.test_data = self.test_data.reshape((picnum, 3, pixel, pixel))
            self.test_data = self.test_data.transpose((0, 2, 3, 1))  # convert to HWC

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


class WarmupLRWrapper:
    """Takes an existing optimizer with corresponding scheduler and warms-up the learning rate."""

    def __init__(self, optimizer, scheduler, warmup_steps):
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.warmup_steps = warmup_steps
        self.current_step = 0

        # Simulate the scheduler for warmup_steps many steps
        for _ in range(self.warmup_steps):
            self.scheduler.step()

        # Get the end lr
        self.end_lr = [pg['lr'] for pg in self.optimizer.param_groups]

        # Set initial lr to a small starting value (i.e. 1/2 of the first actual warmup value)
        for pg_idx, param_group in enumerate(self.optimizer.param_groups):
            param_group['lr'] = self.end_lr[pg_idx] * 0.5 / self.warmup_steps

    def step(self):
        self.current_step += 1
        if self.current_step <= self.warmup_steps:
            # Set the lr to the warmup value (from 0 to self.end_lr)
            for pg_idx, param_group in enumerate(self.optimizer.param_groups):
                param_group['lr'] = self.end_lr[pg_idx] * self.current_step / self.warmup_steps
        else:
            self.scheduler.step()


class SequentialSchedulers(torch.optim.lr_scheduler.SequentialLR):
    """
    Repairs SequentialLR to properly use the last learning rate of the previous scheduler when reaching milestones
    """

    def __init__(self, **kwargs):
        self.optimizer = kwargs['schedulers'][0].optimizer
        super(SequentialSchedulers, self).__init__(**kwargs)

    def step(self):
        self.last_epoch += 1
        idx = bisect_right(self._milestones, self.last_epoch)
        self._schedulers[idx].step()
