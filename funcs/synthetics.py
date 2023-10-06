import numpy as np
from numpy import random
import jax.numpy as jnp
import math

class Quadratic:
    def __init__(self, div=1, n_funcs=5, dim=300):
        alpha = [1.0 / n_funcs for _ in range(n_funcs)]
        coeffs = 1.0 + div * (random.dirichlet(alpha=alpha, size=[dim, ]) - 1.0 / n_funcs)
        coeffs_ = 1.0 + div * (random.dirichlet(alpha=alpha, size=[dim, ]) - 1.0 / n_funcs)
        
        self.fs = []
        for i in range(n_funcs):
            self.fs += [
                lambda x, c1=coeffs[:, i], c2=coeffs_[:, i]: 0.1 / dim * (jnp.dot(c1, self.project(x) ** 2) + jnp.dot(c2, self.project(x)) + 1)
            ]
        self.ws = [1.0 / n_funcs for _ in range(n_funcs)]
        
        self.lb = -10 * np.ones(dim)
        self.ub =  10 * np.ones(dim)
        self.x0 = random.uniform(size=[dim, ], low=0, high=1)
        
    def project(self, x, MIN=0, MAX=1):
        x = np.clip(x, a_min=MIN, a_max=MAX)
        x = x / (MAX - MIN) * (self.ub - self.lb) + self.lb
        return x

class Ackley:
    def __init__(self, div=1, n_funcs=5, dim=300):
        alpha = [1.0 / n_funcs for _ in range(n_funcs)]
        coeffs = 0.1 + div * (random.dirichlet(alpha=alpha, size=[dim, ]) - 1.0 / n_funcs)
        centers = (div * (random.dirichlet(alpha=alpha, size=[dim,]) - 1.0 / n_funcs))

        a, b, c = 20, 0.2, math.pi
        self.fs = []
    
        def f(x):
            part1 = -a * jnp.exp(-b / math.sqrt(dim) * jnp.linalg.norm(x))
            part2 = -(jnp.exp(jnp.mean(jnp.cos(c * x))))
            return part1 + part2 + a + math.e
            
        for i in range(n_funcs):
            self.fs += [
                lambda x, c1=coeffs[:, i], c2=centers[:, i]: f(jnp.multiply(c1, self.project(x) - c2))
            ]
        self.ws = [1.0 / n_funcs for _ in range(n_funcs)]
        
        self.lb = -10 * np.ones(dim)
        self.ub =  10 * np.ones(dim)
        self.x0 = random.uniform(size=[dim, ], low=0, high=1)
        
    def project(self, x, MIN=0, MAX=1):
        x = np.clip(x, a_min=MIN, a_max=MAX)
        x = x / (MAX - MIN) * (self.ub - self.lb) + self.lb
        return jnp.asarray(x)
    

class Levy:
    def __init__(self, div=1, n_funcs=5, dim=300):
        # alpha = np.exp(random.uniform(size=[n_funcs, ]))
        alpha = [1.0 / n_funcs for _ in range(n_funcs)]
        coeffs = 0.1 + div * (random.dirichlet(alpha=alpha, size=[dim, ]) - 1.0 / n_funcs)
        centers = (div * (random.dirichlet(alpha=alpha, size=[dim,]) - 1.0 / n_funcs))

        self.fs = []
    
        def f(x):
            w = 1 + (x - 1.0) / 4.0
            val = jnp.sin(jnp.pi * w[0]) ** 2 + \
                jnp.sum((w[1:dim - 1] - 1) ** 2 * (1 + 10 * jnp.sin(jnp.pi * w[1:dim - 1] + 1) ** 2)) + \
                (w[dim - 1] - 1) ** 2 * (1 + jnp.sin(2 * jnp.pi * w[dim - 1])**2)
            return val
            
        for i in range(n_funcs):
            self.fs += [
                lambda x, c1=coeffs[:, i], c2=centers[:, i]: f(jnp.multiply(c1, self.project(x) - c2))
            ]
        self.ws = [1.0 / n_funcs for _ in range(n_funcs)]
        
        self.lb = -10 * np.ones(dim)
        self.ub =  10 * np.ones(dim)
        self.x0 = random.uniform(size=[dim, ], low=0, high=1)
        
    def project(self, x, MIN=0, MAX=1):
        x = np.clip(x, a_min=MIN, a_max=MAX)
        x = x / (MAX - MIN) * (self.ub - self.lb) + self.lb
        return jnp.asarray(x)