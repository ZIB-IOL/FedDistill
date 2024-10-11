# ===========================================================================
# Project:      On the Byzantine-Resilience of Distillation-Based Federated Learning - IOL Lab @ ZIB
# Paper:        arxiv.org/abs/2402.12265
# File:         actors.py
# Description:  Actor classes for clients and server
# ===========================================================================
import importlib
import os
from collections import OrderedDict
from typing import Optional

import torch
from torchmetrics import MeanMetric
from torchmetrics.classification import MulticlassAccuracy as Accuracy

from utilities import WarmupLRWrapper, SequentialSchedulers


#### Actor Base Class
class Actor:
    """Actor base class"""
    actor_type = None  # To be set by inheriting class

    def __init__(self, use_amp, **kwargs):
        self.use_amp = use_amp

        self.n_classes = kwargs['n_classes']
        self.tmp_dir = kwargs['tmp_dir']
        self.num_workers = kwargs['num_workers']
        self.config = kwargs['config']
        self.callbacks = kwargs['callbacks']
        self.device = kwargs['device']


        # Variables to be set
        self.optimizer = None
        self.scheduler = None
        self.model = None
        self.actor_name = None

        # Define metrics
        self.metrics = {mode: {'loss': MeanMetric().to(device=self.device),
                               'accuracy': Accuracy(num_classes=self.n_classes).to(device=self.device)}
                        for mode in ['train', 'val', 'test']}

        # Define training necessities
        self.gradScaler = torch.cuda.amp.GradScaler(enabled=(self.use_amp is True))
        self.loss_criterion = torch.nn.CrossEntropyLoss(reduction='mean').to(device=self.device)

    def reset_averaged_metrics(self):
        """Resets the metrics
        """
        for mode in self.metrics.keys():
            for metric in self.metrics[mode].values():
                metric.reset()

    def reset_val_and_test_metrics(self):
        """Resets the metrics, but only val and test."""
        for mode in self.metrics.keys():
            if mode in ['val', 'test']:
                for metric in self.metrics[mode].values():
                    metric.reset()

    @torch.no_grad()
    def update_batch_metrics(self, mode: str, loss: torch.tensor, output: torch.tensor, y_target: torch.tensor):
        """Updates metrics given a single batch.

        Args:
            mode (str): train, val or test
            loss (torch.tensor): Single entry tensor with loss of batch
            output (torch.tensor): Output of the model
            y_target (torch.tensor): Target labels
        """
        self.metrics[mode]['loss'](value=loss, weight=output.shape[0])
        if y_target is not None:
            # Otherwise we are tracking on the public dataset
            self.metrics[mode]['accuracy'](output, y_target)

    def get_metrics(self) -> dict:
        """Collects metrics of actor and returns them as a dictionary

        Returns:
            dict: contains metrics of actor 
        """
        with torch.no_grad():
            loggingDict = dict(
                train={metric_name: metric.compute() for metric_name, metric in self.metrics['train'].items() if
                       getattr(metric, 'mode', True) is not None},  # Check if metric computable
                val={metric_name: metric.compute() for metric_name, metric in self.metrics['val'].items()},
                learning_rate=float(self.optimizer.param_groups[0]['lr']),
            )

            loggingDict['test'] = dict()
            for metric_name, metric in self.metrics['test'].items():
                try:
                    # Catch case where MeanMetric mode not set yet
                    loggingDict['test'][metric_name] = metric.compute()
                except Exception:
                    # Reset the test loggingDict, some values are apparently missing
                    loggingDict['test'] = dict()
                    break



        return loggingDict

    def set_model(self, reinit: bool, fileName: Optional[str] = None):
        """Loads model and assigns self.model. If reinit is True, the model architecture is reinitialized.

        Args:
            reinit (bool): If true, then the model architecture is reinitialized.
            fileName (Optional[str], optional): Name of state_dict file in tmp dir, if None then just use the random initialization
        """

        if reinit:
            # Define the model
            model = getattr(importlib.import_module('models.' + self.config.dataset), self.config.arch)()
        else:
            # The model has been initialized already
            model = self.model

        if fileName is not None:
            fPath = os.path.join(self.tmp_dir, fileName)

            state_dict = torch.load(fPath, map_location=self.device)

            new_state_dict = OrderedDict()
            require_DP_format = isinstance(model,
                                           torch.nn.DataParallel)  # If true, ensure all keys start with "module."
            for k, v in state_dict.items():
                is_in_DP_format = k.startswith("module.")
                if require_DP_format and is_in_DP_format:
                    name = k
                elif require_DP_format and not is_in_DP_format:
                    name = "module." + k  # Add 'module' prefix
                elif not require_DP_format and is_in_DP_format:
                    name = k[7:]  # Remove 'module.'
                elif not require_DP_format and not is_in_DP_format:
                    name = k

                v_new = v  # Remains unchanged if not in _orig format
                if k.endswith("_orig"):
                    # We loaded the _orig tensor and corresponding mask
                    name = name[:-5]  # Truncate the "_orig"
                    if f"{k[:-5]}_mask" in state_dict.keys():
                        v_new = v * state_dict[f"{k[:-5]}_mask"]

                new_state_dict[name] = v_new

            maskKeys = [k for k in new_state_dict.keys() if k.endswith("_mask")]
            for k in maskKeys:
                del new_state_dict[k]

            # Load the state_dict
            model.load_state_dict(new_state_dict)
        self.model = model.to(device=self.device)


class Client(Actor):
    """Client class."""
    actor_type = 'client'

    def __init__(self, use_amp, client_id, **kwargs):
        super().__init__(use_amp=use_amp, **kwargs)
        self.client_id = client_id
        self.actor_name = f'client-{self.client_id}'

        # Define private variables which are to be set
        self.trainData = None
        self.dataloader = None
        self.model = None
        self.original_loss = None
        self.is_byzantine = False

        # Checkpoint/Early Stopping variables
        self.best_checkpoint_model = None
        self.best_checkpoint_val_accuracy = 0

    def assign_dataset(self, trainData: torch.utils.data.Subset):
        """Assigns dataset, creates dataloader
        Args:
            trainData (torch.utils.data.Subset): Private Training dataset of client
        """
        self.trainData = trainData
        self.dataloader = torch.utils.data.DataLoader(trainData, batch_size=self.config.batch_size, shuffle=True,
                                                      pin_memory=torch.cuda.is_available(),
                                                      num_workers=self.num_workers)

    def load_checkpoint(self):
        """Loads the checkpoint of the client."""
        # Take the self.best_checkpoint_model and load it
        if self.best_checkpoint_model is not None:
            # Move all tensors to GPU
            self.best_checkpoint_model = {key: val.to(device=self.device) for key, val in
                                          self.best_checkpoint_model.items()}
            # Load the state dict directly from self.best_checkpoint_model
            self.model.load_state_dict(self.best_checkpoint_model)
            self.model = self.model.to(device=self.device)

            del self.best_checkpoint_model
            self.best_checkpoint_model = None
            self.best_checkpoint_val_accuracy = 0

    def update_checkpoint(self):
        """Updates the checkpoint of the client."""
        # Get the current validation accuracy
        val_accuracy = self.metrics['val']['accuracy'].compute()
        if val_accuracy >= self.best_checkpoint_val_accuracy:
            self.best_checkpoint_val_accuracy = val_accuracy

            # Delete the old checkpoint model if existing
            if self.best_checkpoint_model is not None:
                del self.best_checkpoint_model

            # Save the state dict directly to self.best_checkpoint_model, copying the tensors and moving to CPU
            self.best_checkpoint_model = {key: val.detach().clone().cpu() for key, val in
                                          self.model.state_dict().items()}

    def detach_model(self):
        """Detach the model to avoid OOM, i.e., we save the state dict and reload it when needed.
        """
        pass

    def attach_model(self):
        """Re-attach the model, i.e., we reload the state dict.
        """
        pass

    def save_model(self, modelType: str) -> str:
        """Saves current model to os.path.join(self.tmp_dir, f"{modelType}_model.pt"), returns the complete file path.

        Args:
            modelType (str): Name of model type such as 'initial'.

        Returns:
            str: Absolute path to saved model state dict.
        """

        fName = f"{modelType}_model.pt"
        fPath = os.path.join(self.tmp_dir, fName)

        # Only save models in their non-module version, to avoid problems when loading
        try:
            model_state_dict = self.model.module.state_dict()
        except AttributeError:
            model_state_dict = self.model.state_dict()

        torch.save(model_state_dict, fPath)  # Save the state_dict
        return fPath

    def set_optimizer_and_scheduler(self, n_epochs: int, n_batches_per_epoch: int, reinit_optimizer: bool = True):
        """
        Sets the optimizer and scheduler.
        Args:
            n_epochs (int): Use the specified amount of epochs for the learning rate.
            n_batches_per_epoch (int): Number of batches per epoch.
            reinit_optimizer (bool): If True, reinit the optimizer, otherwise keep.
        """
        if self.actor_type == 'client':
            learning_rate = self.config.client_lr
        elif self.actor_type == 'server':
            learning_rate = self.config.server_lr
        else:
            raise NotImplementedError(f"Actor type {self.actor_type} not implemented.")
        
        self.define_optimizer_scheduler(learning_rate=learning_rate, n_epochs=n_epochs,
                                        n_batches_per_epoch=n_batches_per_epoch, reinit_optimizer=reinit_optimizer,
                                        do_warmup=False)

    def define_optimizer_scheduler(self, learning_rate: str, n_epochs: int, n_batches_per_epoch: int,
                                   do_warmup: bool = False, reinit_optimizer: bool = True):
        """
        Defines optimizer and learning rate scheduler, sets self.optimizer and self.scheduler.
        Args:
            learning_rate (str): Learning rate schedule in the form of (type, kwargs)
            n_epochs (int): Number of epochs to run scheduler for
            n_batches_per_epoch (int): Number of batches per epoch
            do_warmup (bool): If True, warmup for 5% of iterations
            reinit_optimizer (bool): If True, reinit the optimizer, otherwise keep.
        """
        # Learning rate scheduler in the form (type, kwargs)
        tupleStr = learning_rate.strip()
        # Remove parenthesis
        if tupleStr[0] == '(':
            tupleStr = tupleStr[1:]
        if tupleStr[-1] == ')':
            tupleStr = tupleStr[:-1]
        name, *kwargs = tupleStr.split(',')
        if name in ['StepLR', 'MultiStepLR', 'ExponentialLR', 'Linear', 'Cosine', 'Constant']:
            scheduler = (name, kwargs)
            initial_lr = float(kwargs[0])
        else:
            raise NotImplementedError(f"LR Scheduler {name} not implemented.")

        # Define the optimizer
        wd = self.config['weight_decay'] or 0.
        if reinit_optimizer:
            if self.config.optimizer == 'SGD':
                self.optimizer = torch.optim.SGD(params=self.model.parameters(), lr=initial_lr,
                                                 momentum=self.config.momentum,
                                                 weight_decay=wd, nesterov=wd > 0.)
            elif self.config.optimizer == 'AdamW':
                self.optimizer = torch.optim.AdamW(params=self.model.parameters(), lr=initial_lr, weight_decay=wd)
            else:
                raise NotImplementedError("Only SGD and AdamW implemented at the moment.")

        # We define a scheduler. All schedulers work on a per-iteration basis
        iterations_per_epoch = n_batches_per_epoch
        n_total_iterations = iterations_per_epoch * n_epochs
        n_warmup_iterations = 0

        # Set the initial learning rate
        for param_group in self.optimizer.param_groups: param_group['lr'] = initial_lr

        # Define the warmup scheduler if needed
        warmup_scheduler, milestone = None, None
        if do_warmup and int(0.05 * n_total_iterations) > 0:
            n_warmup_iterations = int(0.05 * n_total_iterations)
            # As a start factor we use 1e-20, to avoid division by zero when putting 0.
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=self.optimizer,
                                                                 start_factor=1e-20, end_factor=1.,
                                                                 total_iters=n_warmup_iterations)
            milestone = n_warmup_iterations + 1

        n_remaining_iterations = n_total_iterations - n_warmup_iterations

        name, kwargs = scheduler
        scheduler = None
        if name == 'Constant':
            scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer=self.optimizer,
                                                            factor=1.0,
                                                            total_iters=n_remaining_iterations)
        elif name == 'StepLR':
            # Tuple of form ('StepLR', initial_lr, step_size, gamma)
            # Reduces initial_lr by gamma every step_size epochs
            step_size, gamma = int(kwargs[1]), float(kwargs[2])

            # Convert to iterations
            step_size = iterations_per_epoch * step_size

            scheduler = torch.optim.lr_scheduler.StepLR(optimizer=self.optimizer, step_size=step_size,
                                                        gamma=gamma)
        elif name == 'MultiStepLR':
            # Tuple of form ('MultiStepLR', initial_lr, milestones, gamma)
            # Reduces initial_lr by gamma every epoch that is in the list milestones
            milestones, gamma = kwargs[1].strip(), float(kwargs[2])
            # Remove square bracket
            if milestones[0] == '[':
                milestones = milestones[1:]
            if milestones[-1] == ']':
                milestones = milestones[:-1]
            # Convert to iterations directly
            milestones = [int(ms) * iterations_per_epoch for ms in milestones.split('|')]
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=self.optimizer, milestones=milestones,
                                                             gamma=gamma)
        elif name == 'ExponentialLR':
            # Tuple of form ('ExponentialLR', initial_lr, gamma)
            gamma = float(kwargs[1])
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=gamma)
        elif name in ['Linear']:
            if len(kwargs) == 2:
                # The final learning rate has also been passed
                end_factor = float(kwargs[1]) / float(initial_lr)
            else:
                end_factor = 0.
            scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=self.optimizer,
                                                          start_factor=1.0, end_factor=end_factor,
                                                          total_iters=n_remaining_iterations)

        elif name == 'Cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                                   T_max=n_remaining_iterations, eta_min=0.)

        # Reset base lrs to make this work
        scheduler.base_lrs = [initial_lr if warmup_scheduler else 0. for _ in self.optimizer.param_groups]

        # Define the Sequential Scheduler
        if warmup_scheduler is None:
            self.scheduler = scheduler
        elif name in ['StepLR', 'MultiStepLR']:
            # We need parallel schedulers, since the steps should be counted during warmup
            self.scheduler = torch.optim.lr_scheduler.ChainedScheduler(schedulers=[warmup_scheduler, scheduler])
        else:
            self.scheduler = SequentialSchedulers(optimizer=self.optimizer, schedulers=[warmup_scheduler, scheduler],
                                                  milestones=[milestone])

    def warmup_scheduler(self, warmup_steps: int):
        """Adds a short warmup of the learning rate to the current scheduler."""
        if warmup_steps > 0:
            self.scheduler = WarmupLRWrapper(
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                warmup_steps=warmup_steps)


class Server(Client):
    """Server class."""
    actor_type = 'server'

    def __init__(self, use_amp, **kwargs):
        super().__init__(use_amp=use_amp, client_id=None, **kwargs)
        self.actor_name = 'server'
