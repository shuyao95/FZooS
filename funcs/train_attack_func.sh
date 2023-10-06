CUDA_VISIBLE_DEVICES=0 python train_mnist.py --portion 1.0 --save-model &
CUDA_VISIBLE_DEVICES=1 python train_mnist.py --portion 0.9 --save-model &
CUDA_VISIBLE_DEVICES=2 python train_mnist.py --portion 0.8 --save-model &
CUDA_VISIBLE_DEVICES=3 python train_mnist.py --portion 0.7 --save-model &

CUDA_VISIBLE_DEVICES=4 python train_mnist.py --portion 0.6 --save-model &
CUDA_VISIBLE_DEVICES=5 python train_mnist.py --portion 0.5 --save-model &
CUDA_VISIBLE_DEVICES=6 python train_mnist.py --portion 0.4 --save-model &
CUDA_VISIBLE_DEVICES=0 python train_mnist.py --portion 0.3 --save-model &

CUDA_VISIBLE_DEVICES=1 python train_mnist.py --portion 0.2 --save-model &
CUDA_VISIBLE_DEVICES=2 python train_mnist.py --portion 0.1 --save-model &


CUDA_VISIBLE_DEVICES=3 python train_cifar10.py --portion 1.0 &
CUDA_VISIBLE_DEVICES=4 python train_cifar10.py --portion 0.9 &
CUDA_VISIBLE_DEVICES=5 python train_cifar10.py --portion 0.8 &
CUDA_VISIBLE_DEVICES=6 python train_cifar10.py --portion 0.7 &

CUDA_VISIBLE_DEVICES=0 python train_cifar10.py --portion 0.6 &
CUDA_VISIBLE_DEVICES=1 python train_cifar10.py --portion 0.5 &
CUDA_VISIBLE_DEVICES=2 python train_cifar10.py --portion 0.4 &
CUDA_VISIBLE_DEVICES=3 python train_cifar10.py --portion 0.3 &

CUDA_VISIBLE_DEVICES=4 python train_cifar10.py --portion 0.2 &
CUDA_VISIBLE_DEVICES=5 python train_cifar10.py --portion 0.1 &
