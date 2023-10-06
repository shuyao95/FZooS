import numpy as np
import jax.numpy as jnp
from copy import deepcopy
from scipy.optimize import approx_fprime
from multiprocessing import Process
from multiprocessing.pool import ThreadPool as Pool
from optimizers import RFFGP
import time
from tqdm import tqdm

from jax import grad

def aggre_grad(gps, ws, x):
    f = lambda x: sum([w * gp.predict(x.reshape(1, -1)).item() for w, gp in zip(ws, gps)])
    gx = approx_fprime(x, f, epsilon=1e-20)
    return gx
    
def get_rff(opt, f, x, x_r=None):
    opt.queries += [(x, f(x))]
    
    if x_r is not None:
        xs_exp = opt.explore(x, r=x_r, num=200)
    else:
        xs_exp = opt.explore(x, r=0.02, num=200)
        
    xs_base, _ = opt.local_queries(target_x=x, max_queries=150)
    xs_exp = opt.minimize_uncertainty(xs_base, xs_exp, n_exp=50)
    opt.queries += [(x, f(x)) for x in xs_exp]
    opt.get_rff(target_x=x, max_queries=500)
    return deepcopy(opt.rff_gp)


class FedZoo:
    def __init__(self, fs, ws, opts):
        self.fs = fs # individual functions / local functions in clients
        self.ws = ws # weights for individual functions
        # the global function that needs to be optimized
        self.F = lambda x: sum([w*f(x) for w,f in zip(ws, fs)])
        
        self.opts = opts
        
        self.dF_sur = None
        self.gps = [None for _ in self.fs]
        
        self.diff_surs = [None for _ in self.fs]
        self.rff_gps = [None for _ in self.fs]
    
    @staticmethod
    def run_client(opt, f, gp, x, x_prev=None, F=None, dF_sur=None, diff_surs=None, rff_gp=None, args=None):
        opt.dF_sur = dF_sur
        opt.dF = grad(F) if F is not None else None
        
        if dF_sur is not None:
            if args.correction == 'none':
                opt.df_sur = lambda z: None
            elif args.correction == 'true':
                opt.df_sur = lambda z: grad(f)(x)
            elif args.correction == 'prox':
                opt.df_sur = lambda z: x_prev
            elif args.correction == 'diff':
                opt.df_sur = lambda z: diff_surs
            elif args.correction == 'scaf':
                df_sur = opt.gx_avg 
                opt.df_sur = lambda z: df_sur
            elif args.correction == 'post':
                df_sur = aggre_grad([gp], [1], x)
                opt.df_sur = lambda z: aggre_grad([gp], [1], z)
            elif args.correction == 'rff':
                opt.df_sur = lambda z: rff_gp.grad_mean(z)
            
        x = opt.update(f, x, args.iters)
        return x
    
    def run_server(self, xs, x, args):
        prev_x = x
        
        dx = sum([w * (x - xi) for w, xi in zip(self.ws, xs)])
        x = x - args.eta * dx
        if self.F(prev_x) <= self.F(x):
            x = prev_x
        
        x_rs = [np.mean(np.abs(x - xi)) / 2  for xi in xs]
        print(x_rs)
        
        if args.correction == 'none':
            self.dF_sur = None
        elif args.correction == 'true':
            self.dF_sur = lambda z: grad(self.F)(x)
        elif args.correction == 'prox':
            self.dF_sur = lambda z: z # fixed regularizer with 1
        elif args.correction == 'diff':
            for i, opt in enumerate(self.opts):
                self.diff_surs[i] = opt.rand_grad_est(self.fs[i], x)[0]
                
            dF_sur = sum([
                w * df_sur for w, df_sur in zip(self.ws, self.diff_surs)
            ])
            self.dF_sur = lambda z: dF_sur
        elif args.correction == 'scaf':
            dF_sur = sum([w * opt.gx_avg for w, opt in zip(self.ws, self.opts)])
            self.dF_sur = lambda z: dF_sur
        elif args.correction == 'post':
            for i, opt in enumerate(self.opts):
                xs_exp = opt.explore(x, r=0.01, num=200)
                xs_base = opt.fit_gp(target_x=x, max_queries=100)
                xs_exp = opt.minimize_uncertainty(xs_base, xs_exp, n_exp=20)
                
                opt.queries += [(x, self.fs[i](x)) for x in xs_exp]
                opt.fit_gp(target_x=x, max_queries=150)
            
            self.gps = [deepcopy(opt.gp) for opt in self.opts]
            self.dF_sur = lambda z: aggre_grad(self.gps, self.ws, z)
        elif args.correction == 'rff':
            self.rff_gps = []
            with Pool(len(self.opts)) as p:
                self.rff_gps = [
                    p.apply(
                        get_rff, 
                        args=(self.opts[i], self.fs[i], x, x_rs[i])
                    ) for i in range(len(self.opts))
                ]
                
            self.gps = [deepcopy(opt.gp) for opt in self.opts]
                
            self.RFF_GP = RFFGP(
                dim=x.shape[-1], lengthscale=self.rff_gps[0].lengthscale, 
                n_components=self.rff_gps[0].n_components
            )
            self.RFF_GP.build_features()
            self.RFF_GP.nu_mean = sum([
                w * rff_gp.nu_mean * (rff_gp.ys_max - rff_gp.ys_min) \
                    for w, rff_gp in zip(self.ws, self.rff_gps)
            ])
            self.RFF_GP.nu_std = sum([
                w * rff_gp.nu_std * (rff_gp.ys_max - rff_gp.ys_min) \
                    for w, rff_gp in zip(self.ws, self.rff_gps)
            ])
            self.RFF_GP.ys_min = sum([
                w * rff_gp.ys_min for w, rff_gp in zip(self.ws, self.rff_gps)
            ])
            self.RFF_GP.ys_max = self.RFF_GP.ys_min + 1
            self.dF_sur = lambda z: self.RFF_GP.grad_mean(z) # lambda z: self.RFF_GP.grad_rsd(z)
            
        return x
    
    def run(self, x, args, exp=None):
        history = []
        
        if 'zoos' in args.zo_opt:
            for opt, f in zip(self.opts, self.fs):
                xs = opt.explore(x, r=0.01, num=args.n_inits)
                opt.queries += [(x, f(x)) for x in xs]
        
        for r in tqdm(range(args.rounds), leave=False):
            
            x_prev = x
            with Pool(len(self.fs)) as p:
                xs = [
                    p.apply(
                        FedZoo.run_client, 
                        args=(
                            self.opts[i], self.fs[i], self.gps[i], x, x_prev, 
                            self.F, self.dF_sur, self.diff_surs[i], self.rff_gps[i], args,
                        )
                    ) for i in range(len(self.fs))
                ]

            x = self.run_server(xs, x, args)
            num_queries = sum([len(opt.queries) for opt in self.opts])
            if args.success_rate:
                history += [(num_queries, exp.F_success(x))]
                print("Round %03d, Queries %03d, Value %.6f, Success %.6f" %(r+1, num_queries, self.F(x), exp.F_success(x)))
            else:
                history += [(num_queries, self.F(x))]
                print("Round %03d, Queries %03d, Value %.6f" %(r+1, num_queries, self.F(x)))
        
        return history
    