import os
num_thread = 10
os.environ["OMP_NUM_THREADS"] = str(num_thread)
os.environ["OPENBLAS_NUM_THREADS"] = str(num_thread)
os.environ["MKL_NUM_THREADS"] = str(num_thread)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(num_thread)
os.environ["NUMEXPR_NUM_THREADS"] = str(num_thread)
os.environ["BLIS_NUM_THREADS"] = str(num_thread)
os.environ["XLA_FLAGS"] = ("--xla_cpu_multi_thread_eigen=false "
                           "intra_op_parallelism_threads={}".format(num_thread))

from settings import *
from optimizers import *
from funcs.adversarial import *
from funcs.opt_metric import *
from funcs.synthetics import *
from tqdm import tqdm
import argparse
import pickle
import traceback
import sys

torch.set_num_threads(num_thread)



def parse_arguments():
    parser = argparse.ArgumentParser(description='FZooS experiments')
    
    # for synthetic functions
    parser.add_argument('--exp', default='quadratic', type=str, help='[quadratic, adversarial, tuning]')
    parser.add_argument('--dim', type=int, default=300, help='dimention of synthetic functions')
    parser.add_argument('--n_funcs', type=int, default=5, help='number of clients/functions')
    parser.add_argument('--div', type=float, default=1, help='standard deviation of different functions')
    parser.add_argument('--portion', type=float, default=1.0, help='portion of classes in each client')
    parser.add_argument('--ckpts', type=str, default='/Volumes/ZSSD/Research/Attack/ckpts/1.0', help='path of checkpoints')
    
    # for federated zeroth-order optimization
    parser.add_argument('--zo_opt', default='fzoos', type=str, help='type of zeroth-order optimization, [fzoos, rgf, gd]')
    parser.add_argument('--fo_opt', default='adam', type=str, help='type of first-order optimization, [sgd, adam, adgrad]')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate for local updates')
    parser.add_argument('--eta', default=1.0, type=float, help='learning rate for global updates')
    parser.add_argument('--rounds', type=int, default=50, help='number of rounds for communication')
    parser.add_argument('--iters', type=int, default=30, help='number of iterations per round')
    parser.add_argument('--correction', type=str, default="post", help='[none, true, diff, scaf, post, rff]')
    
    # for finite difference-based algorithm
    parser.add_argument('--mu', default=0.01, type=float, help='lenthscale of input perturbation')
    parser.add_argument('--q', default=20, type=int, help='number of queries/pertubations for every gradient estimation')
    
    # for FZooS
    parser.add_argument('--n_inits', type=int, default=10, help='number of initial points for fzoos')
    
    # others
    parser.add_argument('--trials', type=int, default=5, help='number of initial points')
    parser.add_argument('--save', type=str, default="./save", help='filename to save')
    parser.add_argument('--seed', type=int, default=1234, help='random seed')
    parser.add_argument('--score_name', type=str, default='precision_score', help='score function')
    parser.add_argument('--epsilon', type=float, default=1e-20, help='epsilon when using approxi grad')
    parser.add_argument('--success_rate', type=int, default=0, help='whether to use success rate as metric')
    parser.add_argument('--targeted', type=int, default=1, help='whether to targeted attack or not')
    parser.add_argument('--samedata', type=int, default=0, help='whether to use the same data to attack each trial')

    parser.add_argument('--idx', type=int, default=1, help='tmp args')
    
    args = parser.parse_args()
    
    return args
    

if __name__ == '__main__':
    args = parse_arguments()
    folder = os.path.join(args.save, args.exp)
    if not os.path.exists(folder):
        os.makedirs(folder)
        
    data = []
    # while valid_trial < args.trials:
    for j in tqdm(range(args.trials)):
        np.random.seed(args.seed + j)
    
        if 'Attack' in args.exp:
            exp = eval(args.exp)(dir=args.ckpts, idx=args.idx, targeted=args.targeted, samedata=args.samedata)
            args.n_funcs = len(exp.fs)
            
            path = os.path.join(
                folder,
                f"hyper({args.portion},{args.iters})-{args.zo_opt}({args.correction})-{args.samedata}.p",
            )
        elif 'Metric' in args.exp:
            exp = eval(args.exp)(dim=args.dim, portion=args.portion, score_name=args.score_name)
            args.n_funcs = len(exp.fs)
            
            path = os.path.join(
                folder,
                f"{args.score_name}({args.dim},{args.iters},{args.portion})-{args.zo_opt}({args.correction}).p",
            )
        else:
            exp = eval(args.exp)(div=args.div, n_funcs=args.n_funcs, dim=args.dim)
            path = os.path.join(
                folder,
                f"hyper({args.dim},{args.iters},{args.div})-{args.zo_opt}({args.correction}).p",
            )
        
        fo_opt = eval('optax.'+args.fo_opt)(learning_rate=args.lr)
        if args.zo_opt == 'gd':
            zo_opts = [
                gd_opt(fo_opt=fo_opt) for _ in range(args.n_funcs)
            ]
        elif args.zo_opt == 'zoos':
            zo_opts = [
                zoos_opt(fo_opt=fo_opt) for _ in range(args.n_funcs)
            ]
        elif args.zo_opt == 'fzoos':
            zo_opts = [
                fzoos_opt(fo_opt=fo_opt, epsilon=args.epsilon) for _ in range(args.n_funcs)
            ]
        elif args.zo_opt == 'rgf':
            zo_opts = [
                rgf_opt(fo_opt=fo_opt, mu=args.mu, q=args.q) for _ in range(args.n_funcs)
            ]
        elif args.zo_opt == 'prgf':
            zo_opts = [
                prgf_opt(fo_opt=fo_opt, mu=args.mu, q=args.q) for _ in range(args.n_funcs)
            ]
        else:
            raise ValueError('Do not support optimizer [%s]' % args.zo_opt)
        
        # Traceback log
        try:
            problem = FedZoo(exp.fs, exp.ws, opts=zo_opts)
            history = problem.run(exp.x0, args, exp)
            if history:
                data += [history]
            else:
                continue
        except Exception:
            with open(os.path.join(
                folder,
                f"log-{args.iters}-{args.portion}-{args.zo_opt}({args.correction}).txt",
            ), "w") as log:
                traceback.print_exc(file=log)
                sys.exit(1)

    with open(path, 'wb') as f:
        pickle.dump(data, f)

