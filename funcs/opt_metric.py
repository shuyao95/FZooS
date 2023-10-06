import numpy as np
import copy

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, jaccard_score, recall_score, precision_score
from sklearn.preprocessing import StandardScaler 
from sklearn.neural_network import MLPClassifier

from funcs.split_data import *

class MetricOpt:
    def __init__(self, dim=None, lb=-0.2, ub=0.2, portion=1.0, score_name='precision_score'):
        dataset = datasets.fetch_covtype()
        X_all = dataset.data
        y_all = dataset.target
        
        hidden_layer_sizes = (30, 14)
        
        self.dim = dim
        self.lb = lb * np.ones(self.dim)
        self.ub = ub * np.ones(self.dim)
        self.x0 = np.ones(shape=[dim, ]) / 2

        scaler = StandardScaler()  
        scaler.fit(X_all) 

        X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.5, random_state=0)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        
        Xs, ys = split_by_class(X_test, y_test, num_classes=7)
        Xs_split, Ys_split = generate_noniid_data(Xs, ys, 
            portion=portion, num_classes=7
        )
        
        with open('funcs/tuning/ckpts/MetricOpt.npy', 'rb') as f: # load params
            self.params = np.load(f)
        score_func = lambda data, target, model: eval(score_name)(
            target, model.predict(data), average='macro'
        )
            
        self.fs = []
        for X, y in zip(Xs_split, Ys_split):
            clf = MLPClassifier(
                solver='lbfgs', alpha=1e-5, hidden_layer_sizes=hidden_layer_sizes, 
                max_iter=10, random_state=1
            )
            clf.fit(X_train, y_train)
            self._reset_params(clf, np.ones(dim) / 2) 
            
            score = score_func(X, y, clf)
            print('init error', 1 - score)

            def f(x, clf, X, y):
                self._reset_params(clf, x)
                return 1 - score_func(X, y, clf)
            
            self.fs += [
                lambda x, clf=clf, X=X, y=y: f(x, clf, X, y)
            ]
            
        self.ws = [1.0 / len(self.fs) for _ in self.fs]
            
        
    def _reset_params(self, clf, x, MIN=0, MAX=1):
        x = np.clip(x, a_min=MIN, a_max=MAX)
        x = x / (MAX - MIN) * (self.ub - self.lb) + self.lb
        
        
        params = copy.deepcopy(self.params)
        params[-self.dim:] += x
        clf._unpack(params)
    
    
        