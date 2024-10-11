# ===========================================================================
# Project:      On the Byzantine-Resilience of Distillation-Based Federated Learning - IOL Lab @ ZIB
# Paper:        arxiv.org/abs/2402.12265
# File:         main.py
# Description:  Starts up a run
# ===========================================================================

import getpass
import os
import shutil
import socket
import sys
import tempfile
import warnings
from contextlib import contextmanager

import torch
import wandb

from runners.ScratchRunner import ScratchRunner
from utilities import Utilities as Utils
import platform

warnings.filterwarnings('ignore')
debug = "--debug" in sys.argv

defaults = dict(
    # System
    run_id=1,

    # Setup
    dataset='cifar10',
    arch='ResNet18',
    batch_size=128,

    # Efficiency
    use_amp=None,  # Defaults to True

    # Optimizer
    optimizer='SGD',
    momentum=0.9,
    weight_decay=0.0001,
    client_early_stopping=None,
    server_early_stopping=None,

    # Strategy
    strategy='FedAVG',
    n_clients=3,

    # FL settings
    n_total_local_epochs=4,  # Number of local epochs per client (in total!)
    n_communications=3,  # Number of communications between server and clients
    n_server_epochs_per_round=1,  # How many epochs should the server model train per round
    server_lr='(Linear, 0.1)',
    client_lr='(Linear, 0.1)',
    restart_client_lr=None,  # If True, restart the learning rate after each communication
    reinit_server=None,  # Reinitialize server model, optimizer, scheduler after up-communication.
    warm_restarts=None,
    public_ds=None,  # If None, use the default public dataset, otherwise specify the name
    public_ds_fraction=None,  # If None, use the default public dataset, otherwise take fraction of the train set for the public ds

    # Attacks and defences
    defence=None,
    attack='ParameterRandomVectorScaled',
    n_byzantine_clients=None,
    filter_threshold=None,
    filter_quantile=None,
    memory_method=None,  # 'expweights', 'cumsum' or 'quantile'
    expweights=None,
    exp_stepsize=None,
    hips=None,
    sample_attack_frac=None,  # percentage of datapoints on which to choose hips byzantine pred
)

if not debug:
    # Set everything to None recursively
    defaults = Utils.fill_dict_with_none(defaults)

# Add the hostname to the defaults
defaults['computer'] = socket.gethostname()

# Configure wandb logging
wandb.init(
    config=defaults,
    project='test-000',  # automatically changed in sweep
    entity=None,  # automatically changed in sweep
)
config = wandb.config
config = Utils.update_config_with_default(config, defaults)
n_gpus = torch.cuda.device_count()
if n_gpus > 0:
    config.update(dict(device='cuda:0'))
else:
    config.update(dict(device='cpu'))


@contextmanager
def tempdir():
    path = tempfile.mkdtemp()
    try:
        yield path
    finally:
        try:
            shutil.rmtree(path)
            sys.stdout.write(f"Removed temporary directory {path}.\n")
        except IOError:
            sys.stderr.write('Failed to clean up temp dir {}'.format(path))


with tempdir() as tmp_dir:    
    sys.stdout.write(f"Using config: {config}.\n")
    runner = ScratchRunner(config=config, tmp_dir=tmp_dir, debug=debug)

    runner.run()

    # Close wandb run
    wandb_dir_path = wandb.run.dir
    wandb.join()

    # Delete the local files
    if os.path.exists(wandb_dir_path):
        shutil.rmtree(wandb_dir_path)
