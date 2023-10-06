# Federated Zeroth-order Optimization using Trajectory-Informed Surrogate Gradients

This repository contains the code for the experiments of the paper: Federated Zeroth-order Optimization using Trajectory-Informed Surrogate Gradients. 

## 1. Preparing for the environment
Running the following command to install the required packages using the `environment.yml` file using conda:
```
conda env create -f environment.yml
```

## 2. Parameters for the script
The main script is in the file `run_exp.py`. The parameters are as follows:
```
--exp: the name of the experiment
--dim: the input dimensions of the function
--zo_opt: the type of zeroth-order optimization, [fzoos, rgf, gd]
--correction: the name of the optimization algorithm, [none, true, diff, scaf, post, rff]
--div: the standard deviation of different functions
--iters: the number of local iterations per round
--rounds: the number of rounds for communication
--ckpts: the path of checkpoints
--portion: the portion of classes in training data used to train each local model
--n_funcs: the number of functions/clients in the federated learning
--trials: the number of trials
--score_name: the name of the metric unsed in the non-differentiable metrics optimization
--targeted: whether to conduct targeted attack or not, [0,1]
```
## 3. Commands to run the experiments
### 3.1 Synthetic experiments
```
python run_exp.py --zo_opt rgf --correction none --exp Quadratic --div 0.5 --iters 10 --rounds 50
python run_exp.py --zo_opt rgf --correction diff --exp Quadratic --div 0.5 --iters 10 --rounds 50
python run_exp.py --zo_opt rgf --correction prox --exp Quadratic --div 0.5 --iters 10 --rounds 50
python run_exp.py --zo_opt rgf --correction scaf --exp Quadratic --div 0.5 --iters 10 --rounds 50
python run_exp.py --zo_opt fzoos --correction rff --exp Quadratic --div 0.5 --iters 10 --rounds 50


python run_exp.py --zo_opt rgf --correction none --exp Quadratic --div 5 --iters 10 --rounds 50
python run_exp.py --zo_opt rgf --correction diff --exp Quadratic --div 5 --iters 10 --rounds 50
python run_exp.py --zo_opt rgf --correction prox --exp Quadratic --div 5 --iters 10 --rounds 50
python run_exp.py --zo_opt rgf --correction scaf --exp Quadratic --div 5 --iters 10 --rounds 50
python run_exp.py --zo_opt fzoos --correction rff --exp Quadratic --div 5 --iters 10 --rounds 50


python run_exp.py --zo_opt rgf --correction none --exp Quadratic --div 50 --iters 20 --rounds 50
python run_exp.py --zo_opt rgf --correction diff --exp Quadratic --div 50 --iters 20 --rounds 50
python run_exp.py --zo_opt rgf --correction prox --exp Quadratic --div 50 --iters 20 --rounds 50
python run_exp.py --zo_opt rgf --correction scaf --exp Quadratic --div 50 --iters 20 --rounds 50
python run_exp.py --zo_opt fzoos --correction rff --exp Quadratic --div 50 --iters 20 --rounds 50


python run_exp.py --zo_opt rgf --correction none --exp Quadratic --div 5 --iters 5 --rounds 50
python run_exp.py --zo_opt rgf --correction diff --exp Quadratic --div 5 --iters 5 --rounds 50
python run_exp.py --zo_opt rgf --correction prox --exp Quadratic --div 5 --iters 5 --rounds 50
python run_exp.py --zo_opt rgf --correction scaf --exp Quadratic --div 5 --iters 5 --rounds 50
python run_exp.py --zo_opt fzoos --correction rff --exp Quadratic --div 5 --iters 5 --rounds 50


python run_exp.py --zo_opt rgf --correction none --exp Quadratic --div 5 --iters 20 --rounds 50
python run_exp.py --zo_opt rgf --correction diff --exp Quadratic --div 5 --iters 20 --rounds 50
python run_exp.py --zo_opt rgf --correction prox --exp Quadratic --div 5 --iters 20 --rounds 50
python run_exp.py --zo_opt rgf --correction scaf --exp Quadratic --div 5 --iters 20 --rounds 50
python run_exp.py --zo_opt fzoos --correction rff --exp Quadratic --div 5 --iters 20 --rounds 50


python run_exp.py --zo_opt rgf --correction none --exp Levy --div 0.5 --iters 5 --rounds 50
python run_exp.py --zo_opt rgf --correction diff --exp Levy --div 0.5 --iters 5 --rounds 50
python run_exp.py --zo_opt rgf --correction prox --exp Levy --div 0.5 --iters 5 --rounds 50
python run_exp.py --zo_opt rgf --correction scaf --exp Levy --div 0.5 --iters 5 --rounds 50
python run_exp.py --zo_opt fzoos --correction rff --exp Levy --div 0.5 --iters 5 --rounds 50


python run_exp.py --zo_opt rgf --correction none --exp Ackley --div 0.5 --iters 5 --rounds 50
python run_exp.py --zo_opt rgf --correction diff --exp Ackley --div 0.5 --iters 5 --rounds 50
python run_exp.py --zo_opt rgf --correction prox --exp Ackley --div 0.5 --iters 5 --rounds 50
python run_exp.py --zo_opt rgf --correction scaf --exp Ackley --div 0.5 --iters 5 --rounds 50
python run_exp.py --zo_opt fzoos --correction rff --exp Ackley --div 0.5 --iters 5 --rounds 50
```

### 3.2 Federated black-box adversarial attack
Before we run experiments to attack the models, we need to train the model first. After running the above command, we get the resultant model stored in the folder `./funcs/attack/ckpts`. Then we can run the following command to attack the models.

#### a) CIFAR-10
```
CUDA_VISIBLE_DEVICES="" python run_exp.py --exp CIFAR10_Attack --n_funcs 10 --dim 1024 --zo_opt rgf --correction none --ckpts 'funcs/attack/ckpts/0.5' --portion 0.5 --iters 10 --trials 15 --rounds 150 --targeted 0
CUDA_VISIBLE_DEVICES="" python run_exp.py --exp CIFAR10_Attack --n_funcs 10 --dim 1024 --zo_opt rgf --correction diff --ckpts 'funcs/attack/ckpts/0.5' --portion 0.5 --iters 10 --trials 15 --rounds 150 --targeted 0
CUDA_VISIBLE_DEVICES="" python run_exp.py --exp CIFAR10_Attack --n_funcs 10 --dim 1024 --zo_opt rgf --correction prox --ckpts 'funcs/attack/ckpts/0.5' --portion 0.5 --iters 10 --trials 15 --rounds 150 --targeted 0
CUDA_VISIBLE_DEVICES="" python run_exp.py --exp CIFAR10_Attack --n_funcs 10 --dim 1024 --zo_opt rgf --correction scaf --ckpts 'funcs/attack/ckpts/0.5' --portion 0.5 --iters 10 --trials 15 --rounds 150 --targeted 0
CUDA_VISIBLE_DEVICES="" python run_exp.py --exp CIFAR10_Attack --n_funcs 10 --dim 1024 --zo_opt fzoos --correction rff --ckpts 'funcs/attack/ckpts/0.5' --portion 0.5 --iters 10 --trials 15 --rounds 150 --targeted 0

CUDA_VISIBLE_DEVICES="" python run_exp.py --exp CIFAR10_Attack --n_funcs 10 --dim 1024 --zo_opt rgf --correction none --ckpts 'funcs/attack/ckpts/0.7' --portion 0.7 --iters 5 --trials 15 --rounds 150 --targeted 0
CUDA_VISIBLE_DEVICES="" python run_exp.py --exp CIFAR10_Attack --n_funcs 10 --dim 1024 --zo_opt rgf --correction diff --ckpts 'funcs/attack/ckpts/0.7' --portion 0.7 --iters 5 --trials 15 --rounds 150 --targeted 0
CUDA_VISIBLE_DEVICES="" python run_exp.py --exp CIFAR10_Attack --n_funcs 10 --dim 1024 --zo_opt rgf --correction prox --ckpts 'funcs/attack/ckpts/0.7' --portion 0.7 --iters 5 --trials 15 --rounds 150 --targeted 0
CUDA_VISIBLE_DEVICES="" python run_exp.py --exp CIFAR10_Attack --n_funcs 10 --dim 1024 --zo_opt rgf --correction scaf --ckpts 'funcs/attack/ckpts/0.7' --portion 0.7 --iters 5 --trials 15 --rounds 150 --targeted 0
CUDA_VISIBLE_DEVICES="" python run_exp.py --exp CIFAR10_Attack --n_funcs 10 --dim 1024 --zo_opt fzoos --correction rff --ckpts 'funcs/attack/ckpts/0.7' --portion 0.7 --iters 5 --trials 15 --rounds 150 --targeted 0

CUDA_VISIBLE_DEVICES="" python run_exp.py --exp CIFAR10_Attack --n_funcs 10 --dim 1024 --zo_opt rgf --correction none --ckpts 'funcs/attack/ckpts/0.7' --portion 0.7 --iters 10 --trials 15 --rounds 150 --targeted 0
CUDA_VISIBLE_DEVICES="" python run_exp.py --exp CIFAR10_Attack --n_funcs 10 --dim 1024 --zo_opt rgf --correction diff --ckpts 'funcs/attack/ckpts/0.7' --portion 0.7 --iters 10 --trials 15 --rounds 150 --targeted 0
CUDA_VISIBLE_DEVICES="" python run_exp.py --exp CIFAR10_Attack --n_funcs 10 --dim 1024 --zo_opt rgf --correction prox --ckpts 'funcs/attack/ckpts/0.7' --portion 0.7 --iters 10 --trials 15 --rounds 150 --targeted 0
CUDA_VISIBLE_DEVICES="" python run_exp.py --exp CIFAR10_Attack --n_funcs 10 --dim 1024 --zo_opt rgf --correction scaf --ckpts 'funcs/attack/ckpts/0.7' --portion 0.7 --iters 10 --trials 15 --rounds 150 --targeted 0
CUDA_VISIBLE_DEVICES="" python run_exp.py --exp CIFAR10_Attack --n_funcs 10 --dim 1024 --zo_opt fzoos --correction rff --ckpts 'funcs/attack/ckpts/0.7' --portion 0.7 --iters 10 --trials 15 --rounds 150 --targeted 0

CUDA_VISIBLE_DEVICES="" python run_exp.py --exp CIFAR10_Attack --n_funcs 10 --dim 1024 --zo_opt rgf --correction none --ckpts 'funcs/attack/ckpts/0.7' --portion 0.7 --iters 20 --trials 15 --rounds 150 --targeted 0
CUDA_VISIBLE_DEVICES="" python run_exp.py --exp CIFAR10_Attack --n_funcs 10 --dim 1024 --zo_opt rgf --correction diff --ckpts 'funcs/attack/ckpts/0.7' --portion 0.7 --iters 20 --trials 15 --rounds 150 --targeted 0
CUDA_VISIBLE_DEVICES="" python run_exp.py --exp CIFAR10_Attack --n_funcs 10 --dim 1024 --zo_opt rgf --correction prox --ckpts 'funcs/attack/ckpts/0.7' --portion 0.7 --iters 20 --trials 15 --rounds 150 --targeted 0
CUDA_VISIBLE_DEVICES="" python run_exp.py --exp CIFAR10_Attack --n_funcs 10 --dim 1024 --zo_opt rgf --correction scaf --ckpts 'funcs/attack/ckpts/0.7' --portion 0.7 --iters 20 --trials 15 --rounds 150 --targeted 0
CUDA_VISIBLE_DEVICES="" python run_exp.py --exp CIFAR10_Attack --n_funcs 10 --dim 1024 --zo_opt fzoos --correction rff --ckpts 'funcs/attack/ckpts/0.7' --portion 0.7 --iters 20 --trials 15 --rounds 150 --targeted 0

CUDA_VISIBLE_DEVICES="" python run_exp.py --exp CIFAR10_Attack --n_funcs 10 --dim 1024 --zo_opt rgf --correction none --ckpts 'funcs/attack/ckpts/0.9' --portion 0.9 --iters 10 --trials 15 --rounds 150 --targeted 0
CUDA_VISIBLE_DEVICES="" python run_exp.py --exp CIFAR10_Attack --n_funcs 10 --dim 1024 --zo_opt rgf --correction diff --ckpts 'funcs/attack/ckpts/0.9' --portion 0.9 --iters 10 --trials 15 --rounds 150 --targeted 0
CUDA_VISIBLE_DEVICES="" python run_exp.py --exp CIFAR10_Attack --n_funcs 10 --dim 1024 --zo_opt rgf --correction prox --ckpts 'funcs/attack/ckpts/0.9' --portion 0.9 --iters 10 --trials 15 --rounds 150 --targeted 0
CUDA_VISIBLE_DEVICES="" python run_exp.py --exp CIFAR10_Attack --n_funcs 10 --dim 1024 --zo_opt rgf --correction scaf --ckpts 'funcs/attack/ckpts/0.9' --portion 0.9 --iters 10 --trials 15 --rounds 150 --targeted 0
CUDA_VISIBLE_DEVICES="" python run_exp.py --exp CIFAR10_Attack --n_funcs 10 --dim 1024 --zo_opt fzoos --correction rff --ckpts 'funcs/attack/ckpts/0.9' --portion 0.9 --iters 10 --trials 15 --rounds 150 --targeted 0
```
#### b) MNIST
```
CUDA_VISIBLE_DEVICES="" python run_exp.py --exp MNIST_Attack --n_funcs 10 --dim 784 --zo_opt rgf --correction none --ckpts 'funcs/attack/ckpts/0.7' --portion 0.7 --iters 5 --trials 15 --rounds 100 --targeted 0
CUDA_VISIBLE_DEVICES="" python run_exp.py --exp MNIST_Attack --n_funcs 10 --dim 784 --zo_opt rgf --correction diff --ckpts 'funcs/attack/ckpts/0.7' --portion 0.7 --iters 5 --trials 15 --rounds 100 --targeted 0
CUDA_VISIBLE_DEVICES="" python run_exp.py --exp MNIST_Attack --n_funcs 10 --dim 784 --zo_opt rgf --correction prox --ckpts 'funcs/attack/ckpts/0.7' --portion 0.7 --iters 5 --trials 15 --rounds 100 --targeted 0
CUDA_VISIBLE_DEVICES="" python run_exp.py --exp MNIST_Attack --n_funcs 10 --dim 784 --zo_opt rgf --correction scaf --ckpts 'funcs/attack/ckpts/0.7' --portion 0.7 --iters 5 --trials 15 --rounds 100 --targeted 0
CUDA_VISIBLE_DEVICES="" python run_exp.py --exp MNIST_Attack --n_funcs 10 --dim 784 --zo_opt fzoos --correction rff --ckpts 'funcs/attack/ckpts/0.7' --portion 0.7 --iters 5 --trials 15 --rounds 100 --targeted 0

CUDA_VISIBLE_DEVICES="" python run_exp.py --exp MNIST_Attack --n_funcs 10 --dim 784 --zo_opt rgf --correction none --ckpts 'funcs/attack/ckpts/0.7' --portion 0.7 --iters 10 --trials 15 --rounds 100 --targeted 0
CUDA_VISIBLE_DEVICES="" python run_exp.py --exp MNIST_Attack --n_funcs 10 --dim 784 --zo_opt rgf --correction diff --ckpts 'funcs/attack/ckpts/0.7' --portion 0.7 --iters 10 --trials 15 --rounds 100 --targeted 0
CUDA_VISIBLE_DEVICES="" python run_exp.py --exp MNIST_Attack --n_funcs 10 --dim 784 --zo_opt rgf --correction prox --ckpts 'funcs/attack/ckpts/0.7' --portion 0.7 --iters 10 --trials 15 --rounds 100 --targeted 0
CUDA_VISIBLE_DEVICES="" python run_exp.py --exp MNIST_Attack --n_funcs 10 --dim 784 --zo_opt rgf --correction scaf --ckpts 'funcs/attack/ckpts/0.7' --portion 0.7 --iters 10 --trials 15 --rounds 100 --targeted 0
CUDA_VISIBLE_DEVICES="" python run_exp.py --exp MNIST_Attack --n_funcs 10 --dim 784 --zo_opt fzoos --correction rff --ckpts 'funcs/attack/ckpts/0.7' --portion 0.7 --iters 10 --trials 15 --rounds 100 --targeted 0

CUDA_VISIBLE_DEVICES="" python run_exp.py --exp MNIST_Attack --n_funcs 10 --dim 784 --zo_opt rgf --correction none --ckpts 'funcs/attack/ckpts/0.7' --portion 0.7 --iters 20 --trials 15 --rounds 100 --targeted 0
CUDA_VISIBLE_DEVICES="" python run_exp.py --exp MNIST_Attack --n_funcs 10 --dim 784 --zo_opt rgf --correction diff --ckpts 'funcs/attack/ckpts/0.7' --portion 0.7 --iters 20 --trials 15 --rounds 100 --targeted 0
CUDA_VISIBLE_DEVICES="" python run_exp.py --exp MNIST_Attack --n_funcs 10 --dim 784 --zo_opt rgf --correction prox --ckpts 'funcs/attack/ckpts/0.7' --portion 0.7 --iters 20 --trials 15 --rounds 100 --targeted 0
CUDA_VISIBLE_DEVICES="" python run_exp.py --exp MNIST_Attack --n_funcs 10 --dim 784 --zo_opt rgf --correction scaf --ckpts 'funcs/attack/ckpts/0.7' --portion 0.7 --iters 20 --trials 15 --rounds 100 --targeted 0
CUDA_VISIBLE_DEVICES="" python run_exp.py --exp MNIST_Attack --n_funcs 10 --dim 784 --zo_opt fzoos --correction rff --ckpts 'funcs/attack/ckpts/0.7' --portion 0.7 --iters 20 --trials 15 --rounds 100 --targeted 0

CUDA_VISIBLE_DEVICES="" python run_exp.py --exp MNIST_Attack --n_funcs 10 --dim 784 --zo_opt rgf --correction none --ckpts 'funcs/attack/ckpts/0.5' --portion 0.5 --iters 10 --trials 15 --rounds 100 --targeted 0
CUDA_VISIBLE_DEVICES="" python run_exp.py --exp MNIST_Attack --n_funcs 10 --dim 784 --zo_opt rgf --correction diff --ckpts 'funcs/attack/ckpts/0.5' --portion 0.5 --iters 10 --trials 15 --rounds 100 --targeted 0
CUDA_VISIBLE_DEVICES="" python run_exp.py --exp MNIST_Attack --n_funcs 10 --dim 784 --zo_opt rgf --correction prox --ckpts 'funcs/attack/ckpts/0.5' --portion 0.5 --iters 10 --trials 15 --rounds 100 --targeted 0
CUDA_VISIBLE_DEVICES="" python run_exp.py --exp MNIST_Attack --n_funcs 10 --dim 784 --zo_opt rgf --correction scaf --ckpts 'funcs/attack/ckpts/0.5' --portion 0.5 --iters 10 --trials 15 --rounds 100 --targeted 0
CUDA_VISIBLE_DEVICES="" python run_exp.py --exp MNIST_Attack --n_funcs 10 --dim 784 --zo_opt fzoos --correction rff --ckpts 'funcs/attack/ckpts/0.5' --portion 0.5 --iters 10 --trials 15 --rounds 100 --targeted 0

CUDA_VISIBLE_DEVICES="" python run_exp.py --exp MNIST_Attack --n_funcs 10 --dim 784 --zo_opt rgf --correction none --ckpts 'funcs/attack/ckpts/0.9' --portion 0.9 --iters 10 --trials 15 --rounds 100 --targeted 0
CUDA_VISIBLE_DEVICES="" python run_exp.py --exp MNIST_Attack --n_funcs 10 --dim 784 --zo_opt rgf --correction diff --ckpts 'funcs/attack/ckpts/0.9' --portion 0.9 --iters 10 --trials 15 --rounds 100 --targeted 0
CUDA_VISIBLE_DEVICES="" python run_exp.py --exp MNIST_Attack --n_funcs 10 --dim 784 --zo_opt rgf --correction prox --ckpts 'funcs/attack/ckpts/0.9' --portion 0.9 --iters 10 --trials 15 --rounds 100 --targeted 0
CUDA_VISIBLE_DEVICES="" python run_exp.py --exp MNIST_Attack --n_funcs 10 --dim 784 --zo_opt rgf --correction scaf --ckpts 'funcs/attack/ckpts/0.9' --portion 0.9 --iters 10 --trials 15 --rounds 100 --targeted 0
CUDA_VISIBLE_DEVICES="" python run_exp.py --exp MNIST_Attack --n_funcs 10 --dim 784 --zo_opt fzoos --correction rff --ckpts 'funcs/attack/ckpts/0.9' --portion 0.9 --iters 10 --trials 15 --rounds 100 --targeted 0
```
### 3.3 Federated non-differentiable metric optimization
```
CUDA_VISIBLE_DEVICES="" python run_exp.py --exp MetricOpt --n_funcs 7 --dim 2187 --zo_opt rgf --correction none --portion 0.3 --iters 10 --trials 5 --score_name precision_score
CUDA_VISIBLE_DEVICES="" python run_exp.py --exp MetricOpt --n_funcs 7 --dim 2187 --zo_opt rgf --correction diff --portion 0.3 --iters 10 --trials 5 --score_name precision_score
CUDA_VISIBLE_DEVICES="" python run_exp.py --exp MetricOpt --n_funcs 7 --dim 2187 --zo_opt rgf --correction prox --portion 0.3 --iters 10 --trials 5 --score_name precision_score
CUDA_VISIBLE_DEVICES="" python run_exp.py --exp MetricOpt --n_funcs 7 --dim 2187 --zo_opt rgf --correction scaf --portion 0.3 --iters 10 --trials 5 --score_name precision_score
CUDA_VISIBLE_DEVICES="" python run_exp.py --exp MetricOpt --n_funcs 7 --dim 2187 --zo_opt fzoos --correction rff --portion 0.3 --iters 10 --trials 5 --score_name precision_score

CUDA_VISIBLE_DEVICES="" python run_exp.py --exp MetricOpt --n_funcs 7 --dim 2187 --zo_opt rgf --correction none --portion 0.6 --iters 5 --trials 5 --score_name precision_score
CUDA_VISIBLE_DEVICES="" python run_exp.py --exp MetricOpt --n_funcs 7 --dim 2187 --zo_opt rgf --correction diff --portion 0.6 --iters 5 --trials 5 --score_name precision_score
CUDA_VISIBLE_DEVICES="" python run_exp.py --exp MetricOpt --n_funcs 7 --dim 2187 --zo_opt rgf --correction prox --portion 0.6 --iters 5 --trials 5 --score_name precision_score
CUDA_VISIBLE_DEVICES="" python run_exp.py --exp MetricOpt --n_funcs 7 --dim 2187 --zo_opt rgf --correction scaf --portion 0.6 --iters 5 --trials 5 --score_name precision_score
CUDA_VISIBLE_DEVICES="" python run_exp.py --exp MetricOpt --n_funcs 7 --dim 2187 --zo_opt fzoos --correction rff --portion 0.6 --iters 5 --trials 5 --score_name precision_score

CUDA_VISIBLE_DEVICES="" python run_exp.py --exp MetricOpt --n_funcs 7 --dim 2187 --zo_opt rgf --correction none --portion 0.6 --iters 10 --trials 5 --score_name precision_score
CUDA_VISIBLE_DEVICES="" python run_exp.py --exp MetricOpt --n_funcs 7 --dim 2187 --zo_opt rgf --correction diff --portion 0.6 --iters 10 --trials 5 --score_name precision_score
CUDA_VISIBLE_DEVICES="" python run_exp.py --exp MetricOpt --n_funcs 7 --dim 2187 --zo_opt rgf --correction prox --portion 0.6 --iters 10 --trials 5 --score_name precision_score
CUDA_VISIBLE_DEVICES="" python run_exp.py --exp MetricOpt --n_funcs 7 --dim 2187 --zo_opt rgf --correction scaf --portion 0.6 --iters 10 --trials 5 --score_name precision_score
CUDA_VISIBLE_DEVICES="" python run_exp.py --exp MetricOpt --n_funcs 7 --dim 2187 --zo_opt fzoos --correction rff --portion 0.6 --iters 10 --trials 5 --score_name precision_score

CUDA_VISIBLE_DEVICES="" python run_exp.py --exp MetricOpt --n_funcs 7 --dim 2187 --zo_opt rgf --correction none --portion 0.6 --iters 20 --trials 5 --score_name precision_score
CUDA_VISIBLE_DEVICES="" python run_exp.py --exp MetricOpt --n_funcs 7 --dim 2187 --zo_opt rgf --correction diff --portion 0.6 --iters 20 --trials 5 --score_name precision_score
CUDA_VISIBLE_DEVICES="" python run_exp.py --exp MetricOpt --n_funcs 7 --dim 2187 --zo_opt rgf --correction prox --portion 0.6 --iters 20 --trials 5 --score_name precision_score
CUDA_VISIBLE_DEVICES="" python run_exp.py --exp MetricOpt --n_funcs 7 --dim 2187 --zo_opt rgf --correction scaf --portion 0.6 --iters 20 --trials 5 --score_name precision_score
CUDA_VISIBLE_DEVICES="" python run_exp.py --exp MetricOpt --n_funcs 7 --dim 2187 --zo_opt fzoos --correction rff --portion 0.6 --iters 20 --trials 5 --score_name precision_score

CUDA_VISIBLE_DEVICES="" python run_exp.py --exp MetricOpt --n_funcs 7 --dim 2187 --zo_opt rgf --correction none --portion 0.9 --iters 10 --trials 5 --score_name precision_score
CUDA_VISIBLE_DEVICES="" python run_exp.py --exp MetricOpt --n_funcs 7 --dim 2187 --zo_opt rgf --correction diff --portion 0.9 --iters 10 --trials 5 --score_name precision_score
CUDA_VISIBLE_DEVICES="" python run_exp.py --exp MetricOpt --n_funcs 7 --dim 2187 --zo_opt rgf --correction prox --portion 0.9 --iters 10 --trials 5 --score_name precision_score
CUDA_VISIBLE_DEVICES="" python run_exp.py --exp MetricOpt --n_funcs 7 --dim 2187 --zo_opt rgf --correction scaf --portion 0.9 --iters 10 --trials 5 --score_name precision_score
CUDA_VISIBLE_DEVICES="" python run_exp.py --exp MetricOpt --n_funcs 7 --dim 2187 --zo_opt fzoos --correction rff --portion 0.9 --iters 10 --trials 5 --score_name precision_score
```
Change the option `--score_name` to `recall_score`, `jaccard_score`, `f1_score` to run the experiments for different metrics.