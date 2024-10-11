## On the Byzantine-Resilience of Distillation-Based Federated Learning

*Authors: [Christophe Roux](http://christopheroux.de/), [Max Zimmer](https://maxzimmer.org/), [Sebastian Pokutta](http://www.pokutta.com/)*

This repository contains the code to reproduce the experiments from the paper ["On the Byzantine-Resilience of Distillation-Based Federated Learning"](https://arxiv.org/abs/2402.12265).
The code is based on [PyTorch 1.9](https://pytorch.org/) and the experiment-tracking platform [Weights & Biases](https://wandb.ai).

### Structure and Usage
#### Structure
Experiments are started from the following file:

- [`main.py`](main.py): Starts experiments using the dictionary format of Weights & Biases.

The rest of the project is structured as follows:

- [`byzantine`](byzantine): Contains the attacks and defenses used in the paper.
- [`runners`](runners): Contains classes to control the training and collection of metrics.
- [`models`](models): Contains all model architectures used.
- [`utilities.py`](utilities.py): Contains useful auxiliary functions and classes.
- [`config.py`](config.py): Configuration for the datasets used in the experiments.
- [`public_config.py`](public_config.py): Contains the configuration for the public datasets.
- [`metrics.py`](metrics.py): Contains the metrics used in the experiments.
- [`strategies.py`](strategies.py): Contains the different strategies used, such as FedAVG and FedDistill.


#### Usage
Define the parameters in the [`main.py`](main.py) defaults-dictionary and run it with the --debug flag. Or, configure a sweep in Weights & Biases and run it from there (without the flag).

### Citation

In case you find the paper or the implementation useful for your own research, please consider citing:

```
@article{roux2024byzantine,
  author = {Roux, Christophe and Zimmer, Max and Pokutta, Sebastian},
  title = {On the Byzantine-Resilience of Distillation-Based Federated Learning},
  year = {2024},
  journal = {arXiv preprint arXiv:2402.12265},
}
```
