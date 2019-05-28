# Deep Variational Koopman Models
Source code for "Deep Variational Koopman Models: Inferring Koopman Observations for Uncertainty-Aware Dynamics Modeling and Control" from IJCAI 2019. The paper can be found [here](https://arxiv.org/pdf/1902.09742.pdf).

## Overview
A description of the individual files is given below.
* ```variational_koopman_model.py``` - script for defining architecture of and constructing Deep Variational Koopman models for training.
* ```train_variational_koopman.py``` - training script for Deep Variational Koopman models.
* ```replay_memory.py``` - script for loading and processing data prior to and during training.
* ```utils.py``` - contains functions for visualizing performance of trained models and performing MPC rollouts.
* ```controller.py``` - script for performing DDP (essentially iLQR) for optimizing action sequences.

### Training and Evaluating a Model
To train a Deep Variational Koopman model on data from the inverted pendulum environment with the model architecture used in the experiments within the paper, run ```train_variational_koopman.py``` with the following arguments:

```python3 train_variational_koopman.py --seq_length 16 --n_trials 20 --n_subseq 220 --kl_weight 0.1 --extractor_size 64 64 --inference_size 64 64 --prior_size 64 32```

This will extract data from 20 trials within the inverted pendulum environment, where each trial is 256 time steps in length and actions are selected randomly. 

To train a model and subsequently incorporate it into MPC, run the above command and include the argument ```--ilqr True```. This will train a model, perform ```n_trials/10``` MPC rollouts using the trained model, then incorporate the data from the rollouts into the training/validation sets and train a new model on the full dataset. This process will terminate once a reward threshold is reached.

To train models on data from different environments, include the argument ```--domain_name (domain_name)```, where ```(domain_name)``` is the name of the associated environment, such as ```--domain_name CartPole-v1```. Currently it is not possible to perform MPC with any environment other than inverted pendulum.
