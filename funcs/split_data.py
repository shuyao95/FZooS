import numpy as np

def split_by_class(X, Y, num_classes=10):
    Xs, Ys = [], []
    for i in range(num_classes):
        Xi = X[Y == i]
        Yi = i * np.ones(len(Xi))
        Xs += [Xi]
        Ys += [Yi]
    return Xs, Ys

def generate_noniid_data(Xs, Ys, portion=1.0, num_classes=10):
    classes_per_client = int(portion * num_classes)
    ids = np.arange(num_classes)
    
    Xs_split = [[] for _ in range(num_classes)]
    Ys_split = [[] for _ in range(num_classes)]
    
    # determine which client each class should go to
    assignment = {i: [] for i in range(num_classes)}
    for _ in range(classes_per_client):
        np.random.shuffle(ids)
        for i in range(num_classes):
            assignment[i] += [ids[i]]
    
    for i in range(num_classes):
        np.random.shuffle(Xs[i])
        num_per_client = len(Xs[i]) // classes_per_client
        parts = [k * num_per_client for k in range(classes_per_client)] + [len(Xs[i])+1]
        for j, c in enumerate(assignment[i]):
            start, end = parts[j], parts[j+1]
            Xs_split[c] += [Xs[i][start:end]]
            Ys_split[c] += [Ys[i][start:end]]
    
    Xs_split = [np.concatenate(xs) for xs in Xs_split]
    Ys_split = [np.concatenate(ys) for ys in Ys_split]
    return Xs_split, Ys_split
