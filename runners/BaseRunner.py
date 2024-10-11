# ===========================================================================
# Project:      On the Byzantine-Resilience of Distillation-Based Federated Learning - IOL Lab @ ZIB
# Paper:        arxiv.org/abs/2402.12265
# File:         runners/BaseRunner.py
# Description:  Base Runner class, all other runners inherit from this one
# ===========================================================================
import os
import shutil
import sys
import time
from typing import Optional
import getpass

import numpy as np
import torch
import wandb
from torch.cuda.amp import autocast
from tqdm.auto import tqdm
import platform
import strategies
from actors import Client, Server, Actor
from config import (datasetDict, n_classesDict, num_workersDict,
                    testTransformDict, trainTransformDict)
from public_config import (public_datasetAssignmentDict, public_trainDataset_dict, public_testDataset_dict)
from utilities import Utilities as Utils


class BaseRunner:
    """Base class for all runners, defines the general functions"""

    def __init__(self, config: dict, tmp_dir: str, debug: bool):
        """Initialization of variables, server and clients.

        Args:
            config (dict): Wandb run config
            debug (bool): If True, use local datasets instead of specific ones on cluster.
        """

        self.config = config
        self.debug = debug

        # Useful variables
        self.tmp_dir = tmp_dir
        sys.stdout.write(f"Using temporary directory {self.tmp_dir}.\n")
        self.num_workers = num_workersDict[self.config.dataset]
        self.n_classes = n_classesDict[self.config.dataset]
        self.use_amp = torch.cuda.is_available() and self.config.use_amp in [True, None, 'None']
        sys.stdout.write(f"Using AMP: {self.use_amp}.\n")

        # Variables to be set
        self.device = None
        self.seed = None
        self.strategy = None
        self.dataloaders_public = {}
        self.ensemble_test_acc = None
        self.total_epochs_completed = 0
        self.total_bytes_communicated = 0
        self.client_epochs_done, self.server_epochs_done = 0, 0
        self.attack, self.defence = None, None
        self.defence_time = None

        # Configure working device (gpu/cpu, cudnn.benchmark)
        self.configure_comp_device()

        # Define the strategy
        self.define_strategy()

        # Verify input
        self.verify_input()

        # Define clients
        # We have n_clients many actors with ids starting from 1
        self.clients = [Client(use_amp=self.use_amp, client_id=client_id, n_classes=self.n_classes, tmp_dir=self.tmp_dir,
                               num_workers=self.num_workers, config=self.config, callbacks=None, device=self.device)
                        for client_id in range(1, self.config.n_clients + 1, 1)]

        # Define the server model
        self.server = Server(use_amp=self.use_amp, n_classes=self.n_classes, tmp_dir=self.tmp_dir,
                             num_workers=self.num_workers, config=self.config, callbacks=None, device=self.device)

        # Split dataset among client
        self.dataset_rootPath = './datasets_pytorch/' + self.config.dataset

    def configure_comp_device(self):
        """Configure working device (gpu/cpu, cudnn.benchmark)
        """
        self.device = torch.device(self.config.device)
        if 'cuda' in self.config.device:
            torch.cuda.set_device(self.device)
        torch.backends.cudnn.benchmark = True  # Benchmarking for efficiency

    def verify_input(self):
        """Verifies input to baseRunner. This function should only check for non-strategy specific things.
         The rest should be checked in the strategy.
        """
        assert self.config.n_clients > 0, "There must be at least one client."
        # assert self.config.strategy in ['Pretrain','FED','FEDSB','FedAVG', 'FedDF'], 'The chosen strategy is not implemented'

        # Verify input to the strategy
        self.strategy.verify_input()

    def set_seed(self):
        """Sets the seed if existing, otherwise generates a new one, sets it and pushes it to Wandb.
        """
        if self.seed is None:
            # Generate a random seed
            self.seed = int((os.getpid() + 1) * time.time()) % 2 ** 32

        wandb.config.update({'seed': self.seed})  # Push the seed to wandb

        # Set a unique random seed
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        # Remark: If you are working with a multi-GPU model, this function is insufficient to get determinism. To seed all GPUs, use manual_seed_all().
        torch.cuda.manual_seed(self.seed)  # This works if CUDA not available

    def assign_dataloaders(self):
        """Load datasets and split to clients, create dataloaders.
        """
        # Load Train and test datasets
        if self.config.dataset in ['cinic10', 'imagenet100','clothing1m']:
            train_dir = os.path.join(self.dataset_rootPath, 'train')
            test_dir = os.path.join(self.dataset_rootPath, 'test')
            trainData = Utils.get_index_dataset(datasetDict[self.config.dataset])(root=train_dir,
                                                                                  transform=trainTransformDict[
                                                                                      self.config.dataset])



            if self.config.dataset == 'cinic10':
                testData = datasetDict[self.config.dataset](root=test_dir,
                                                                                        transform=testTransformDict[
                                                                                            self.config.dataset])
                # Pick 10% of the test data using a fixed random seed and generator
                cut_off = 10000
                testData, _ = torch.utils.data.random_split(testData, [cut_off, len(testData) - cut_off],
                                                            generator=torch.Generator().manual_seed(0))

                # Overwrite the __getitem__ function of the test data to return the index as well -> we need to do this since the SubSet of random_split yields the wrong indices
                class TestDataset(torch.utils.data.Dataset):
                    def __init__(self, dataset):
                        self.dataset = dataset

                    def __getitem__(self, idx):
                        # Overload this to collect the class indices once in a vector, which can then be used in the sampler
                        item = self.dataset.__getitem__(idx)

                        # If the dataset is unlabeled, we just return None as the label
                        if isinstance(item, tuple):
                            # Labels exist
                            image, label = item
                        else:
                            image = item
                            label = None
                        return image, label, idx

                    def __len__(self):
                        return len(self.dataset)

                testData = TestDataset(testData)
            else:
                testData = Utils.get_index_dataset(datasetDict[self.config.dataset])(root=test_dir,
                                                                                        transform=testTransformDict[
                                                                                            self.config.dataset])

        else:
            trainData = Utils.get_index_dataset(datasetDict[self.config.dataset])(root=self.dataset_rootPath,
                                                                                  train=True,
                                                                                  download=True,
                                                                                  transform=trainTransformDict[
                                                                                      self.config.dataset])

            testData = Utils.get_index_dataset(datasetDict[self.config.dataset])(root=self.dataset_rootPath,
                                                                                 train=False,
                                                                                 transform=testTransformDict[
                                                                                     self.config.dataset])


        n_val_samples_public = int(0.05 * len(trainData))  # Split off some validation data from the entire training set
        n_train_samples = len(trainData) - n_val_samples_public
        # If we use a separate public DS, the no. of public samples from the whole dataset is zero
        #

        if self.config.public_ds_fraction not in [None, 'none', 'None']:
            n_train_samples_public = int(self.config.public_ds_fraction * n_train_samples)
            # We have split a fraction of the train dataset to use as a public dataset
            # Reset indices of trainData_public, needed for correct averaging of outputs
        else:
            n_train_samples_public = 0
        n_train_samples_private = n_train_samples - n_train_samples_public

        trainData_private, trainData_public, valData_public = torch.utils.data.random_split(trainData,
                                                                                            [
                                                                                                n_train_samples_private,
                                                                                                n_train_samples_public,
                                                                                                n_val_samples_public],
                                                                                            generator=torch.Generator().manual_seed(
                                                                                                self.seed))

        if self.config.public_ds_fraction not in [None, 'none', 'None']:
            Utils.reset_dataset_subset_indices(dataset=trainData_public)

        # We specify separate public datasets for clients and server (same dataset, potentially different transforms)
        if self.config.public_ds in [None, 'none', 'None']:
            public_ds_name = public_datasetAssignmentDict[self.config.dataset]
        else:
            public_ds_name = self.config.public_ds
        sys.stdout.write(f"Using public dataset: {public_ds_name}.\n")

        trainData = {}
        if self.config.public_ds_fraction not in [None, 'none', 'None']:
            for actor in ['client', 'server']:
                # this code assumes we use augmenations
                trainData[actor] = trainData_public
        else:
            for actor in ['client', 'server']:
                if actor == 'client':
                    public_ds = public_testDataset_dict[public_ds_name]
                elif actor == 'server':
                    public_ds = public_trainDataset_dict[public_ds_name]
                public_ds_rootPath = './datasets_pytorch/' + public_ds_name
                if public_ds_name == 'cinic10':
                    # use the validation split as the public dataset
                    public_ds_rootPath = os.path.join(public_ds_rootPath, 'valid')
                if public_ds_name == 'clothing1m':
                    # use the validation split as the public dataset
                    public_ds_rootPath = os.path.join(self.dataset_rootPath, 'unlabeled')
                initialized_pub_ds = Utils.get_index_dataset(public_ds)(root=public_ds_rootPath)                    
                trainData[actor] = initialized_pub_ds


        # Define the public loaders
        for mode, data in zip(['train', 'train_server', 'val', 'test'],
                              [trainData['client'], trainData['server'], valData_public, testData]):
            shuffle = ('train' in mode)
            self.dataloaders_public[mode] = torch.utils.data.DataLoader(data, batch_size=self.config.batch_size,
                                                                        shuffle=shuffle,
                                                                        pin_memory=torch.cuda.is_available(),
                                                                        num_workers=self.num_workers)

        # Assign the dataloader to the server
        self.server.assign_dataset(trainData=trainData['server'])

        # Split the remaining trainData_private among the clients
        splitFractions = self.config.n_clients * [len(trainData_private) // self.config.n_clients]
        splitFractions[0] += len(trainData_private) % self.config.n_clients  # Remainder goes to first client


        # We do a uniform split
        trainData_private_split = torch.utils.data.random_split(trainData_private, splitFractions,
                                                                generator=torch.Generator().manual_seed(self.seed))

        for client in self.clients:
            client_id = client.client_id
            client_data_split = trainData_private_split[client_id - 1]
            client.assign_dataset(trainData=client_data_split)
            sys.stdout.write(f"Client {client_id} has {len(client_data_split)} samples.\n")

    def define_strategy(self):
        """Defines the training strategy.
        """
        try:
            self.strategy = getattr(strategies, self.config.strategy)(config=self.config, runner_instance=self)
        except AttributeError:
            raise AttributeError(f"Strategy {self.config.strategy} not found.")

    def log_at_round_end(self, round: int, round_n_epochs: int, round_runtime: float):
        """Logs at the very end of a round and definitely commits."""
        # Get metrics involving all clients
        loggingDict = {}

        # Add round metrics
        loggingDict.update({"round": round,
                            "round_runtime": round_runtime,
                            "round_n_epochs": round_n_epochs,
                            "total_epochs_completed": self.total_epochs_completed,
                            "total_bytes_communicated": self.total_bytes_communicated,
                            })
        
        if self.defence_time is not None:
            loggingDict.update({"defence_time": self.defence_time})

        # Add the server metrics (the last epoch has not been committed)
        loggingDict.update({"server": self.server.get_metrics()})

        if self.ensemble_test_acc is not None:
            loggingDict.update({"ensemble_test_acc": self.ensemble_test_acc})

        wandb.log(loggingDict)

    def log_clients_at_epoch_end(self, epoch: int, commit: bool, loggingDict = None):
        """Logs all client information."""
        if loggingDict is None:
            loggingDict = {f"client{client.client_id}": client.get_metrics() for client in self.clients}

            # Log the early stopping best accuracy
            for client in self.clients:
                if self.config.client_early_stopping:
                    loggingDict.update(
                        {f"client{client.client_id}.best_checkpoint_val_acc": client.best_checkpoint_val_accuracy})
        
        loggingDict.update({"client_epoch": epoch})

        wandb.log(loggingDict, commit=commit)

    def log_server(self, epoch: int, commit: bool):
        """Logs the actor to wandb."""
        # Get actor metrics
        loggingDict = {"server": self.server.get_metrics(),
                       "server_epoch": epoch,
                       }

        wandb.log(loggingDict, commit=commit)

    def final_log(self, client: Optional[Client] = None):
        """
        Performs the final evaluation and logging of client
        Args:
            client (Optional[Client]): client to log. If None, do final logging for all clients and the server
        """
        # Recompute accuracy and loss
        sys.stdout.write(f"\nFinal logging.\n")
        actors = [client] if client else [client for client in self.clients] + [self.server]
        for actor in actors:
            actor.reset_averaged_metrics()
            self.evaluate_model(actor=actor, data='val')
            self.evaluate_model(actor=actor, data='test')

            # Update final metrics
            prefix = actor.actor_type if actor.actor_type == 'server' else f"{actor.actor_type}{actor.client_id}"
            for metric_type, val in actor.get_metrics().items():
                wandb.run.summary[f"final.{prefix}.{metric_type}"] = val

    def train_epoch(self, actor: Actor, data: str = 'train', is_training: bool = True, epoch: Optional[int] = None):
        """Train actor for a single epoch. Used also for evaluation.
                Args:
                    actor (Actor): client to train/evaluate.
                    data (str): train, val, or test. If 'train', use private training data of client.
                                Else use public data.
                    is_training (bool): If true, then collect gradients and update the model.
                    epoch (Optional[int]): If not None, print out the current number of the epoch
                """
        assert data in ['train', 'val', 'test']
        assert not (data in ['test', 'val'] and is_training), "Can't train on test/val set."
        if data == 'train':
            loader = actor.dataloader  # Use the private training data of client
            # Note: if this is for some reason called for the server, abort, we don't want to train server on labels
            assert actor.actor_type == 'client' or not is_training, "Can't train server on labeled public train dataset."
        else:
            loader = self.dataloaders_public[data]

        epochStr = f"\nEpoch {epoch} - " if epoch is not None else ""
        sys.stdout.write(
            f"{epochStr}Training {actor.actor_name} on private data:\n") if is_training else sys.stdout.write(
            f"Evaluating {actor.actor_name} on public {data} data:\n")
        with torch.set_grad_enabled(is_training):
            with tqdm(loader, leave=True) as pbar:
                for x_input, y_target, _ in pbar:
                    # Move to CUDA if possible
                    x_input = x_input.to(self.device, non_blocking=True)
                    y_target = y_target.to(self.device, non_blocking=True)
                    actor.optimizer.zero_grad()  # Zero the gradient buffers

                    if is_training:
                        with autocast(enabled=(self.use_amp is True)):
                            output = actor.model.train()(x_input)
                            loss = actor.loss_criterion(output, y_target)

                        actor.gradScaler.scale(loss).backward()  # AMP gradient scaling + Backpropagation
                        actor.gradScaler.step(actor.optimizer)  # Optimization step
                        actor.gradScaler.update()  # Update AMP gradScaler

                        actor.scheduler.step()
                    else:
                        with autocast(enabled=(self.use_amp is True)):
                            # We use train(mode=True) for the training dataset such that we do not get the drop in loss because of running average of BN
                            # Note however that this will change the running stats and consequently also slightly the evaluation of val/eval datasets
                            output = actor.model.train(mode=(data == 'train'))(x_input)
                            loss = actor.loss_criterion(output, y_target)

                    actor.update_batch_metrics(mode=data, loss=loss, output=output, y_target=y_target)

    def evaluate_model(self, actor: Actor, data: str = 'train'):
        """Evaluates the model of client on the given data

        Args:
            actor (Actor): actor to evaluate
            data (str, optional): train, val or test. Defaults to 'train'.
        """
        self.train_epoch(actor=actor, data=data, is_training=False)
