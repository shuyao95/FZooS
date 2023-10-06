from jax import grad, jit
import optax

import GPy
import numpy as np
from numpy.random import uniform, normal
from scipy.optimize import approx_fprime

from sklearn.gaussian_process.kernels import Matern, RBF
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from scipy.spatial.distance import cosine

import warnings
warnings.simplefilter('ignore', UserWarning)
warnings.simplefilter('ignore', RuntimeWarning)

import time

def grad_error(df, df_est, x, eps=1e-12):
    dx = df(x)
    dx_est = df_est(x)
    
    divergence = np.linalg.norm(dx - dx_est)
    simiarity = 1 - cosine(dx, dx_est)
    return divergence, simiarity


############# Define optimizers for zeroth-order optimization ####################

class zoos_opt:
    def __init__(self, fo_opt=None):
        self.queries = []
        
        self.gp = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            alpha=1e-6,
            normalize_y=False,
            n_restarts_optimizer=20,
            random_state=0,
        )
        
        if fo_opt is None:
            self.fo_opt = optax.adam(learning_rate=0.1)
        else:
            self.fo_opt = fo_opt
            
        self.dF_sur = None
        self.df_sur = None
        
        self.gx_avg = 0 # for SCAFFOLD type II
    
    def local_queries(self, target_x, max_queries=150):
        xs = np.array(list(map(lambda q: q[0], self.queries)))
        ys = np.array(list(map(lambda q: q[1].item(), self.queries)))
        if max_queries > 0: # use local queries to estimate the gradient more accurately
            dists = np.array(list(map(lambda x: np.linalg.norm(x - target_x), xs)))
            idx = np.argsort(dists)[:max_queries]
            xs, ys = xs[idx], ys[idx]
        return xs, ys
    
    def fit_gp(self, target_x, max_queries=150):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            xs, ys = self.local_queries(target_x, max_queries)
            
            self.ys_max, self.ys_min= ys.max(), ys.min()
            ys_trans = (ys - self.ys_min) / (self.ys_max - self.ys_min + 1e-20)
            
            self.gp.fit(xs, ys_trans)
        return xs
    
    def grad_mean(self, x):
        f = lambda x: self.gp.predict(x.reshape(1, -1)).item() * (self.ys_max - self.ys_min) + self.ys_min
        gx = approx_fprime(x, f, epsilon=1e-20)
        return gx
    
    def grad_var(self, X, xs, obs_noise=1e-6):
        n, dim = xs.shape
        
        lengthscale = np.exp(self.gp.kernel_.theta)
        kernel = GPy.kern.RBF(input_dim=dim, lengthscale=lengthscale)
        vars = kernel.dK2_dXdX2(xs, xs, 0, 0) \
                - kernel.dK2_dXdX2(xs, X, 0, 0) @ np.linalg.inv(kernel.K(X, X) \
                + obs_noise * np.identity(X.shape[0])) @ kernel.dK2_dXdX2(X, xs, 0, 0)
        return [vars[i,i].item() for i in range(n)]
    
    def minimize_uncertainty(self, X, xs, n_exp=20):
        vars = self.grad_var(X, xs)
        inds = np.argsort(vars)
        return xs[inds[-n_exp:]]
        
    @staticmethod  
    def explore(x, r=1, num=200, use_normal=False):
        if use_normal:
            samples = normal(size=[num, np.size(x)])
        else:
            samples = uniform(size=[num, np.size(x)], low=-1, high=1)
        dxs = samples
        xs = x.reshape(1, -1) + r * dxs
        return xs
    
    
    def update(self, f, x, iters):
        gd_state = self.fo_opt.init(x)
        self.queries += [(x, f(x))]
        
        # for SCAFFOLD type II
        self.gx_avg = 0
        counter = 0
        
        errors = [0, 0, 0, 0]
        for t in range(iters):
            
            xs = self.fit_gp(target_x=x, max_queries=150)
            gx = self.grad_mean(x)
            if t % 10 == 0:
                print(1 - cosine(gx, grad(f)(x)))
            
            if self.dF_sur is not None:
                div, sim = grad_error(self.dF, self.dF_sur, x)
                errors[0] += div
                errors[1] += sim
                
                div, sim = grad_error(grad(f), self.df_sur, x)
                errors[2] += div
                errors[3] += sim
                counter += 1
                gx += self.dF_sur(x) - self.df_sur(x)
            
            self.gx_avg += gx
            
            dx, gd_state = self.fo_opt.update(gx, gd_state) 
            x = optax.apply_updates(x, dx)
            
            self.queries += [(x, f(x))]
            
        self.gx_avg /= iters
        
        counter += 1e-12
        print('dF: div %.2f, sim %.2f' % (errors[0] / counter, errors[1] / counter))
        print('df: div %.2f, sim %.2f' % (errors[2] / counter, errors[3] / counter))
        print('====================================================')
        return x


class gd_opt:
    def __init__(self, fo_opt=None):
        self.queries = []
        
        if fo_opt is None:
            self.fo_opt = optax.adam(learning_rate=0.1)
        else:
            self.fo_opt = fo_opt

    def update(self, f, x, iters):
        gd_state = self.fo_opt.init(x)
        self.queries += [(x, f(x))]
        
        # for SCAFFOLD type II
        self.gx_avg = 0
        counter = 0
        
        errors = [0, 0, 0, 0]
        # gd optimize 
        for t in range(iters):
            gx = grad(f)(x)
            self.gx_avg += gx
            
            if self.dF_sur is not None:
                div, sim = grad_error(self.dF, self.dF_sur, x)
                errors[0] += div
                errors[1] += sim
                div, sim = grad_error(grad(f), self.df_sur, x)
                errors[2] += div
                errors[3] += sim
                counter += 1
                
                gx += self.dF_sur(x) - self.df_sur(x)
            dx, gd_state = self.fo_opt.update(gx, gd_state)
            
            x = optax.apply_updates(x, dx)
            self.queries += [(x, f(x))]
        
        self.gx_avg /= iters
        counter += 1e-12
        print('dF: div %.2f, sim %.2f' % (errors[0] / counter, errors[1] / counter))
        print('df: div %.2f, sim %.2f' % (errors[2] / counter, errors[3] / counter))
        print('====================================================')
            
        return x


class rgf_opt:
    def __init__(self, mu, q, fo_opt=None):
        self.q = q
        self.mu = mu
        self.queries = []
        
        if fo_opt is None:
            self.fo_opt = optax.adam(learning_rate=0.1)
        else:
            self.fo_opt = fo_opt
            
        self.dF_sur = None
        self.df_sur = None
            
    def rand_grad_est(self, f, x):
        dim = len(x)
        samples = np.random.normal(0, 1, size=(self.q, dim))
        # orthogonalization
        orthos = []
        for u in samples:
            for ou in orthos:
                u = u - np.vdot(u, ou) * ou
            u = u / (np.linalg.norm(u, ord=2) + 1e-20)
            orthos.append(u)
            
        fx, gx = f(x), 0
        
        fd_queries = []
        for s in orthos:
            # s = jnp.asarray(s)
            new_x = x + self.mu * s
            new_fx = f(new_x)
            gx += ((new_fx - fx) / self.mu) * s
            fd_queries += [(new_x, new_fx)]
            
        gx /= len(samples)
        return gx, fd_queries
    
    
    def update(self, f, x, iters):
        gd_state = self.fo_opt.init(x)
        self.queries += [(x, f(x))]
        
        # for SCAFFOLD type II
        self.gx_avg = 0
        counter = 0
        
        errors = [0, 0, 0, 0]
        for t in range(iters):
            gx, fd_queries = self.rand_grad_est(f, x)
            self.gx_avg += gx

            if self.dF_sur is not None:
                gx += self.dF_sur(x) - self.df_sur(x)

            dx, gd_state = self.fo_opt.update(gx, gd_state)
            x = optax.apply_updates(x, dx)
            
            self.queries += fd_queries + [(x, f(x))]
                
        self.gx_avg /= iters
        counter += 1e-12
        return x
    
    
class prgf_opt:
    def __init__(self, mu, q, fo_opt=None):
        self.q = q
        self.mu = mu
        self.queries = []
        
        if fo_opt is None:
            self.fo_opt = optax.adam(learning_rate=0.1)
        else:
            self.fo_opt = fo_opt
            
        self.dF_sur = None
        self.df_sur = None
        
        self.prior = None
            
    def rand_grad_est(self, f, x, priors):
        dim = len(x)
        samples = np.random.normal(0, 1, size=(self.q, dim))
        samples = np.vstack([samples, priors])
        # orthogonalization
        orthos = []
        for u in samples:
            for ou in orthos:
                u = u - np.vdot(u, ou) * ou
            u = u / (np.linalg.norm(u, ord=2) + 1e-20)
            orthos.append(u)
            
        fx, gx = f(x), 0
        
        fd_queries = []
        for s in orthos:
            # s = jnp.asarray(s)
            new_x = x + self.mu * s
            new_fx = f(new_x)
            gx += ((new_fx - fx) / self.mu) * s
            fd_queries += [(new_x, new_fx)]
            
        gx /= len(samples)
        return gx, fd_queries
    
    
    def update(self, f, x, iters):
        if self.prior is None:
            self.prior = np.random.normal(0, 1, size=(1, x.shape[-1]))
        
        gd_state = self.fo_opt.init(x)
        self.queries += [(x, f(x))]
        
        # for SCAFFOLD type II
        self.gx_avg = 0
        counter = 0
        
        errors = [0, 0, 0, 0]
        for t in range(iters):
            gx, fd_queries = self.rand_grad_est(f, x,  priors=self.prior)
            self.gx_avg += gx
            
            if self.dF_sur is not None:
                div, sim = grad_error(self.dF, self.dF_sur, x)
                errors[0] += div
                errors[1] += sim
                div, sim = grad_error(grad(f), self.df_sur, x)
                errors[2] += div
                errors[3] += sim
                counter += 1
                
                gx += self.dF_sur(x) - self.df_sur(x)
            
            dx, gd_state = self.fo_opt.update(gx, gd_state)
            x = optax.apply_updates(x, dx)
            
            self.prior = dx.reshape(1, -1)
            
            self.queries += fd_queries + [(x, f(x))]
                
        self.gx_avg /= iters
        counter += 1e-12
        print('dF: div %.2f, sim %.2f' % (errors[0] / counter, errors[1] / counter))
        print('df: div %.2f, sim %.2f' % (errors[2] / counter, errors[3] / counter))
        print('====================================================')
        return x
    
 

class RFFGP:
    def __init__(self, dim, lengthscale=1.0, n_components=10000):
        self.dim = dim
        self.lengthscale = lengthscale
        self.n_components = n_components
        self.nu = None
        
        # self.linear = Ridge(alpha=5e-6, fit_intercept=False)
        self.linear = Ridge(alpha=0, fit_intercept=False)
        
    def build_features(self, lengthscale=None, random_state=0):
        if lengthscale is None:
            gamma = 1 / (2 * self.lengthscale**2)
        else:
            gamma = 1 / (2 * lengthscale**2)
        
        self.rbf_feature = RBFSampler(
            gamma=gamma, n_components=self.n_components, 
            random_state=random_state
        )
        self.rbf_feature.fit(X=np.zeros(shape=(1, self.dim)))
        
    def fit(self, xs, ys, obs_noise_square=1e-5):
        self.ys_mean, self.ys_std= ys.mean(), ys.std()
        self.ys_max, self.ys_min = ys.max(), ys.min()
        
        ys = (ys - self.ys_min) / ((self.ys_max - self.ys_min) + 1e-40)
        
        X = self.rbf_feature.transform(xs)
        try:
            self.linear.fit(X, ys)
        except:
            print(np.any(np.isnan(X)))
            print(np.any(np.isnan(ys)))
            print(np.any(np.isfinite(X)))
            print(np.any(np.isfinite(ys)))
            np.save('X.npy',X)
            np.save('ys.npy',ys)
            raise ValueError(f'Isnan X: {np.any(np.isnan(X))} ys: {np.any(np.isnan(ys))} Isfinite X: {np.any(np.isfinite(X))} ys: {np.any(np.isfinite(ys))}')

        self.nu_mean = self.linear.coef_ # self.nu = np.linalg.lstsq(X, ys, rcond=None)[0]
        self.nu_std = np.sqrt(1 / (1 / obs_noise_square * np.diag(np.dot(X.T, X)) + 1))
        return self.nu_mean
    
    def predict(self, x, nu_mean=None):
        weights = self.nu_mean if nu_mean is None else nu_mean
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        x_trans = self.rbf_feature.transform(x)
        y = np.dot(x_trans, weights) * (self.ys_max - self.ys_min) + self.ys_min
        return y
    
    def grad_mean(self, x, nu=None):
        weights = self.nu_mean if nu is None else nu
        
        rand_ws = self.rbf_feature.random_weights_
        offset = self.rbf_feature.random_offset_
        
        gx = np.matmul(- np.sin(np.dot(x, rand_ws) + offset), np.einsum('i,ij->ij', weights, rand_ws.T))
        gx *= (2.0 / self.n_components) ** 0.5 * (self.ys_max - self.ys_min)
        return gx

    def grad_rsd(self, x):
        nus = np.random.normal(self.nu_mean, scale=self.nu_std, size=(50, len(self.nu_mean)))
        gxs = np.vstack([self.grad_mean(x, nu) for nu in nus])
        
        gxs_mean = gxs.mean(axis=0)
        cor = max([cosine(gx, gxs_mean) for gx in gxs])
        return cor
        
    
    def minimize_var(self, xs_base, xs_to_check, n_exp=20):
        vars = self.grad_var(xs_base, xs_to_check)
        inds = np.argsort(vars)
        return xs_to_check[inds[-n_exp:]]

class fzoos_opt:
    def __init__(self, fo_opt=None, n_components=10000, epsilon=1e-20):
        self.queries = []
        self.n_components = n_components
        
        self.gp = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            alpha=1e-6,
            normalize_y=False,
            n_restarts_optimizer=20,
            random_state=0,
        )
        
        if fo_opt is None:
            self.fo_opt = optax.adam(learning_rate=0.1)
        else:
            self.fo_opt = fo_opt
            
        self.dF_sur = None
        self.df_sur = None
        
        self.gx_avg = 0 # for SCAFFOLD type II
        
        self.epsilon = epsilon # for approxi grad
    
    def local_queries(self, target_x, max_queries=150, x_r=0.1):
        xs = np.array(list(map(lambda q: q[0], self.queries)))
        ys = np.array(list(map(lambda q: q[1].item(), self.queries)))
        if max_queries > 0: # use local queries to estimate the gradient more accurately
            dists = np.array(list(map(lambda x: np.linalg.norm(x - target_x), xs)))
            idx = np.argsort(dists)[:max_queries]
            xs, ys = xs[idx], ys[idx]
        return xs, ys
    
    def fit_gp(self, target_x, max_queries=150):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            xs, ys = self.local_queries(target_x, max_queries)
            
            self.ys_max, self.ys_min= ys.max(), ys.min()
            ys_trans = (ys - self.ys_min) / (self.ys_max - self.ys_min + 1e-20)
            self.gp.fit(xs, ys_trans)
        return xs, ys
    
    def grad_mean(self, x, gp=None):
        if gp is None:
            gp = self.gp
        f = lambda x: gp.predict(x.reshape(1, -1)).item() * (self.ys_max - self.ys_min) + self.ys_min
        gx = approx_fprime(x, f, epsilon=self.epsilon)
        return gx
    
    def grad_var(self, X, xs, obs_noise=1e-6):
        n, dim = xs.shape
        lengthscale = np.exp(self.gp.kernel_.theta)
        
        kernel = GPy.kern.RBF(input_dim=dim, lengthscale=lengthscale)
        vars = kernel.dK2_dXdX2(xs, xs, 0, 0) \
                - kernel.dK2_dXdX2(xs, X, 0, 0) @ np.linalg.inv(kernel.K(X, X) \
                + obs_noise * np.identity(X.shape[0])) @ kernel.dK2_dXdX2(X, xs, 0, 0)
        return [vars[i,i].item() for i in range(n)]
        
    @staticmethod  
    def explore(x, r=1, num=200, use_normal=False):
        if use_normal:
            samples = normal(size=[num, np.size(x)])
        else:
            samples = uniform(size=[num, np.size(x)], low=-1, high=1)
        dxs = samples
        xs = x.reshape(1, -1) + r * dxs
        return xs

    def minimize_uncertainty(self, X, xs, n_exp=20):
        vars = self.grad_var(X, xs)
        inds = np.argsort(vars)
        return xs[inds[-n_exp:]]
    
    def get_rff(self, target_x, max_queries=150, lengthscale=None):
        xs, ys = self.local_queries(target_x, max_queries)

        lengthscale = 1.0
        self.rff_gp = RFFGP(
            dim=target_x.shape[-1], lengthscale=lengthscale, 
            n_components=self.n_components
        )
        self.rff_gp.build_features()
        nu_mean = self.rff_gp.fit(xs, ys)
        return nu_mean
    
    def update(self, f, x, iters):
        gd_state = self.fo_opt.init(x)
        self.queries += [(x, f(x))]
        
        # for SCAFFOLD type II
        self.gx_avg = 0
        counter = 0
        
        errors = [0, 0, 0, 0]
        gamma = 1.0
        for t in range(iters):
            gamma_t = gamma / (t+1) # decay for better performance
            
            xs, _ = self.fit_gp(target_x=x, max_queries=150)
            gx = self.grad_mean(x)
                
            self.gx_avg += gx
            
            if self.dF_sur is not None:
                gx += gamma_t * (self.dF_sur(x) - self.df_sur(x))

            dx, gd_state = self.fo_opt.update(gx, gd_state) 
            x = optax.apply_updates(x, dx)
                
            self.queries += [(x, f(x))]
            
            if t < iters - 1: # exploration at x (sent to server) will be applied later
                xs_exp = self.explore(x, r=0.01, num=100)
                xs_exp = self.minimize_uncertainty(xs + [x], xs_exp, n_exp=5)
                self.queries += [(x, f(x)) for x in xs_exp]
            
        self.gx_avg /= iters
        
        counter += 1e-12
        return x


