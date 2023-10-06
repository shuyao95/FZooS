import numpy as np
import jax.numpy as jnp
import os

import torch
from torchvision import datasets, transforms
from funcs.attack.models import *
# from numpy.random import normal

class MNIST:
    def __init__(self, device, path):
        self.device = device
        self.model = SimpleCNN().to(device)
        self.model.load_state_dict(torch.load(path, map_location=device))
        self.model.eval()

    def get_loss_and_grad(self, image, label, targeted=True, get_grad=True):
        image = image.clone()
        if get_grad:
            image.requires_grad_()
        input_image = (image - 0.1307) / 0.3081
        output = self.model(input_image)
        batch_size = output.shape[0]
        if batch_size > 1:
            label_term = torch.gather(output, 1,label.reshape(-1,1)).squeeze_()
            other = output + 0.0
            other.scatter_(index=label.reshape(-1,1), dim=1, value=-1e8)
            other_term = torch.max(other, dim=1).values
        else:
            label_term = output[:, label]
            other = output + 0.0
            other[:, label] = -1e8
            other_term = torch.max(other, dim=1).values
        if targeted:
            loss = label_term - other_term
        else:
            loss = other_term - label_term
        
        if batch_size > 1:
            loss = torch.mean(loss)
        else:
            loss = torch.squeeze(loss)
        if get_grad:
            loss.backward()
            grad = image.grad
            return loss.detach().cpu().numpy(), grad.detach()
        else:
            return loss.detach().cpu().numpy()

    def get_pred(self, image, logit=False):
        input_image = (image - 0.1307) / 0.3081
        output = self.model(input_image)
        if logit:
            return output
        else:
            return torch.argmax(output)
    

class CIFAR10:
    def __init__(self, device, path):
        self.device = device
        self.model = ResNet18().to(device)
        self.model.load_state_dict(torch.load(path, map_location=device))
        self.model = self.model.double()
        self.model.eval()

    def get_loss_and_grad(self, image, label, targeted=True, get_grad=True):
        image = image.clone()
        if get_grad:
            image.requires_grad_()
        input_image = image
        output = self.model(input_image)
        label_term = output[:, label]
        other = output + 0.0
        other[:, label] = -1e8
        other_term = torch.max(other, dim=1).values
        if targeted:
            loss = label_term - other_term
        else:
            loss = other_term - label_term        
        loss = torch.squeeze(loss)
        if get_grad:
            loss.backward()
            grad = image.grad
            return loss.detach().cpu().numpy(), grad.detach()
        else:
            return loss.detach().cpu().numpy()
        
    def get_pred(self, image):
        input_image = image
        output = self.model(input_image).detach().cpu().numpy().squeeze()
        return np.argmax(output)


class MNIST_Attack:
    def __init__(self, dim=784, lb=-0.3, ub=0.3, dir='/Volumes/ZSSD/Research/Attack/ckpts/1.0', idx=1, targeted=1, samedata=0):
        self.lb = lb * np.ones(dim)
        self.ub = ub * np.ones(dim)

        # self.device = torch.device("cuda")
        device = torch.device("cpu")
        
        dataset = datasets.MNIST('data', download=True, train=False, transform=transforms.ToTensor())
        # self.size = (1, 1, 28, 28)
        self.dim = 28 * 28
        # data, target = dataset[1]

        self.x0 = np.zeros(shape=[dim, ])
        # self.x0 = normal(size=[dim]) *0.01
        
        if os.path.exists(os.path.join(dir, "m-valid.npy")):
            valid_idx = np.load(os.path.join(dir, "m-valid.npy"))
        else:
            self.fs_d = []
            for file in os.listdir(dir):
                if 'mnist' in file:
                    model = MNIST(device, os.path.join(dir, file))
                    self.fs_d += [
                        lambda x, data, target, model=model, device=device, targeted=targeted: \
                            self.eval(model, data, target, x, device, targeted)
                    ]
            self.ws = [1.0 / len(self.fs_d) for _ in self.fs_d]
            tmp_F = lambda x, data, target: sum([w*f(x, data, target) for w,f in zip(self.ws, self.fs_d)])
            from tqdm import tqdm
            valid_idx = []
            
            for i in tqdm(range(len(dataset))):
                data_tmp, target_tmp = dataset[i]
                if tmp_F(self.x0, data_tmp, target_tmp) >= 0:
                    valid_idx.append(i)
            np.save(os.path.join(dir, "m-valid.npy"),valid_idx)
            print(f"Portion: {dir[-3:]}, Valid count: {len(valid_idx)}")

        if samedata:
            idx = valid_idx[1]
        else:
            idx = np.random.choice(valid_idx, 1).item()
        data, target = dataset[idx]
        
        self.fs = []
        for file in os.listdir(dir):
            if 'mnist' in file:
                model = MNIST(device, os.path.join(dir, file))
                self.fs += [
                    lambda x, model=model, data=data, target=target, device=device, targeted=targeted: \
                        self.eval(model, data, target, x, device, targeted)
                ]
        
        self.ws = [1.0 / len(self.fs) for _ in self.fs]

    def eval(self, model, data, target, x, device, targeted=True):
        x = self.project(np.asarray(x))
        # x = self.project(jnp.asarray(x))
        
        new_image = torch.from_numpy(x.reshape(1, 1, 28, 28)).float() + data
        new_image = torch.clamp(new_image, 0, 1)
        
        target_label = (target + 1) % 10
        
        if targeted:
            # target attack
            loss = model.get_loss_and_grad(new_image.to(device), target_label, targeted=True, get_grad=False)
        else:
            # untarget attack
            loss = model.get_loss_and_grad(new_image.to(device), target, targeted=False, get_grad=False)
        
        return - loss
    
    def project(self, x, MIN=0, MAX=1):
        x = np.clip(x, a_min=MIN, a_max=MAX)
        x = x / (MAX - MIN) * (self.ub - self.lb) + self.lb
        return x
    
    
class CIFAR10_Attack:
    def __init__(self, dim=1024, lb=-0.1, ub=0.1, dir='/Volumes/ZSSD/Research/Attack/ckpts/1.0', idx=1, targeted=1, samedata=0):
        self.lb = lb * np.ones(dim)
        self.ub = ub * np.ones(dim)
        
        # self.device = torch.device("cuda")
        device = torch.device("cpu")
        
        mean, std = [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        dataset = datasets.CIFAR10(root='data', train=False, download=True, transform=transform_test)
        # self.size = (1, 1, 32, 32)
        self.dim = 1 * 32 * 32

        self.x0 = np.zeros(shape=[dim, ])

        trans_lb = (np.zeros((1, 3, 32, 32)) - np.array(mean).reshape(1, 3, 1, 1)) / np.array(std).reshape(1, 3, 1, 1)
        trans_ub = (np.ones((1, 3, 32, 32)) - np.array(mean).reshape(1, 3, 1, 1)) / np.array(std).reshape(1, 3, 1, 1)

        if os.path.exists(os.path.join(dir, "c-valid.npy")):
            valid_idx = np.load(os.path.join(dir, "c-valid.npy"))
        else:
            self.fs_d = []
            for file in os.listdir(dir):
                if 'cifar10' in file:
                    model = CIFAR10(device, os.path.join(dir, file))
                    self.fs_d += [
                        lambda x, data, target, model=model, trans_lb=trans_lb, trans_ub=trans_ub, device=device, targeted=targeted: \
                            self.eval(model, data, target, x, trans_lb, trans_ub, device, targeted)
                    ]
            self.ws = [1.0 / len(self.fs_d) for _ in self.fs_d]
            tmp_F = lambda x, data, target: sum([w*f(x, data, target) for w,f in zip(self.ws, self.fs_d)])
            from tqdm import tqdm
            valid_idx = []
            
            for i in tqdm(range(len(dataset))):
                data_tmp, target_tmp = dataset[i]
                if tmp_F(self.x0, data_tmp, target_tmp) >= 0:
                    valid_idx.append(i)
            np.save(os.path.join(dir, "c-valid.npy"),valid_idx)
            print(f"Portion: {dir[-3:]}, Valid count: {len(valid_idx)}")
        
        if samedata:
            idx = valid_idx[1]
        else:
            idx = np.random.choice(valid_idx, 1).item()
        data, target = dataset[idx]
        
        self.fs = []
        for file in os.listdir(dir):
            if 'cifar10' in file:
                model = CIFAR10(device, os.path.join(dir, file))
                self.fs += [
                    lambda x, model=model, data=data, target=target, trans_lb=trans_lb, trans_ub=trans_ub, device=device, targeted=targeted: \
                        self.eval(model, data, target, x, trans_lb, trans_ub, device, targeted)
                ]
        self.ws = [1.0 / len(self.fs) for _ in self.fs]

    def eval(self, model, data, target, x, trans_lb, trans_ub, device, targeted):
        x = self.project(x)
        
        new_image = torch.from_numpy(np.asarray(x).reshape(1, 1, 32, 32)).double() + data
        new_image = torch.clamp(new_image, torch.from_numpy(trans_lb), torch.from_numpy(trans_ub))
        
        target_label = (target + 1) % 10

        if targeted:
            # target attack
            loss = model.get_loss_and_grad(new_image.to(device), target_label, targeted=True, get_grad=False)
        else:
            # untarget attack
            loss = model.get_loss_and_grad(new_image.to(device), target, targeted=False, get_grad=False)
        
        return - loss

    def project(self, x, MIN=0, MAX=1):
        x = np.clip(x, a_min=MIN, a_max=MAX)
        x = x / (MAX - MIN) * (self.ub - self.lb) + self.lb
        return x
    
    

# class MNIST_Attack_Batch:
#     def __init__(self, dim=784, lb=-0.3, ub=0.3, dir='/Volumes/ZSSD/Research/Attack/ckpts/1.0', idx=1):
#         self.batch_size = 15
#         self.lb = lb * np.ones(dim)
#         self.ub = ub * np.ones(dim)
        
#         dataset = datasets.MNIST('data', download=True, train=False, transform=transforms.ToTensor())
#         # self.size = (1, 1, 28, 28)
#         self.dim = 28 * 28
#         # data, target = dataset[1]
#         # data, target = dataset[3]
        
#         # attack from class 3 to 5
        
#         batch = [dataset[i] for i in np.arange(idx, idx+self.batch_size)]
#         data, target = torch.stack([d[0] for d in batch], dim=0), torch.tensor([d[1] for d in batch])
        
#         self.x0 = np.zeros(shape=[dim, ])
#         # self.x0 = normal(size=[dim]) *0.01
        
#         # self.device = torch.device("cuda")
#         device = torch.device("cpu")
#         self.fs = []
#         self.fs_logit = []
        
#         for file in os.listdir(dir):
#             if 'mnist' in file:
#                 model = MNIST(device, os.path.join(dir, file))
#                 self.fs += [
#                     lambda x, model=model, data=data, target=target, device=device: \
#                         self.eval(model, data, target, x, device)
#                 ]
#                 self.fs_logit += [
#                     lambda x, model=model, data=data, device=device: \
#                         self.eval_logit(model, data, x, device)
#                 ]
#         self.ws = [1.0 / len(self.fs) for _ in self.fs]
        
#         F_logit = lambda x: sum([w*f(x) for w,f in zip(self.ws, self.fs_logit)])
#         self.F_success = lambda x, F_logit=F_logit, target=target: self.eval_sucess(F_logit, target, x)
        
#     def eval(self, model, data, target, x, device, targeted=True):
#         x = self.project(np.asarray(x))
#         # x = self.project(jnp.asarray(x))
        
#         new_image = torch.from_numpy(x.reshape(1, 1, 28, 28)).float().repeat([self.batch_size,1,1,1]) + data
#         new_image = torch.clamp(new_image, 0, 1)
        
#         target_label = (target + 1) % 10
        
#         if targeted:
#             # target attack
#             loss = model.get_loss_and_grad(new_image.to(device), target_label, targeted=True, get_grad=False)
#         else:
#             # untarget attack
#             loss = model.get_loss_and_grad(new_image.to(device), target, targeted=False, get_grad=False)
        
#         return - loss
    
#     def eval_logit(self, model, data, x, device):
#         x = self.project(np.asarray(x))
#         new_image = torch.from_numpy(x.reshape(1, 1, 28, 28)).float().repeat([self.batch_size,1,1,1]) + data
#         new_image = torch.clamp(new_image, 0, 1)
#         output = model.get_pred(new_image.to(device), logit=True)
#         return output
    
#     def eval_sucess(self, F_logit, target, x):
#         logit_output = F_logit(x)
        
#         label_term = torch.gather(logit_output, 1,target.reshape(-1,1)).squeeze_()
#         other = logit_output + 0.0
#         other.scatter_(index=target.reshape(-1,1), dim=1, value=-1e8)
#         other_term = torch.max(other, dim=1).values
#         loss = other_term - label_term
#         print("Logits:" + str(loss))
#         return sum(loss > 0).item() / self.batch_size

#     def project(self, x, MIN=0, MAX=1):
#         x = np.clip(x, a_min=MIN, a_max=MAX).reshape(-1)
#         x = x / (MAX - MIN) * (self.ub - self.lb) + self.lb
#         return x
    
    
# class CIFAR10_Attack_Batch:
#     def __init__(self, dim=1024, lb=-0.1, ub=0.1, dir='/Volumes/ZSSD/Research/Attack/ckpts/1.0'):
#         self.lb = lb * np.ones(dim)
#         self.ub = ub * np.ones(dim)
        
#         mean, std = [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]
#         transform_test = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize(mean, std),
#         ])
#         dataset = datasets.CIFAR10(root='data', train=False, download=True, transform=transform_test)
#         # self.size = (1, 1, 32, 32)
#         self.dim = 1 * 32 * 32
        
#         trans_lb = (np.zeros((1, 3, 32, 32)) - np.array(mean).reshape(1, 3, 1, 1)) / np.array(std).reshape(1, 3, 1, 1)
#         trans_ub = (np.ones((1, 3, 32, 32)) - np.array(mean).reshape(1, 3, 1, 1)) / np.array(std).reshape(1, 3, 1, 1)
        
#         self.x0 = np.zeros(shape=[dim, ])
        
#         # self.device = torch.device("cuda")
#         device = torch.device("cpu")
#         data, target = dataset[1]
#         self.fs = []
        
#         for file in os.listdir(dir):
#             if 'cifar10' in file:
#                 model = CIFAR10(device, os.path.join(dir, file))
#                 self.fs += [
#                     lambda x, model=model, data=data, target=target, trans_lb=trans_lb, trans_ub=trans_ub, device=device: \
#                         self.eval(model, data, target, x, trans_lb, trans_ub, device)
#                 ]
#         self.ws = [1.0 / len(self.fs) for _ in self.fs]
        
#     def eval(self, model, data, target, x, trans_lb, trans_ub, device):
#         x = self.project(x)
        
#         new_image = torch.from_numpy(np.asarray(x).reshape(1, 1, 32, 32)).double() + data
#         new_image = torch.clamp(new_image, torch.from_numpy(trans_lb), torch.from_numpy(trans_ub))
        
#         target_label = (target + 1) % 10
        
#         loss = model.get_loss_and_grad(new_image.to(device), target_label, targeted=True, get_grad=False)
        
#         return - loss

#     def project(self, x, MIN=0, MAX=1):
#         x = np.clip(x, a_min=MIN, a_max=MAX)
#         x = x / (MAX - MIN) * (self.ub - self.lb) + self.lb
#         return x