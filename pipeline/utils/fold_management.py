import numpy as np
from sklearn.model_selection import KFold

def KFold_methode(x, number_groups=5, seed=None, shuffle=False):
    kf = KFold(n_splits=number_groups, random_state=seed, shuffle=shuffle)        
    return kf.split(x)

def fold_massive_methode(dict_massives):
    unique_massives = list(dict_massives.keys())
    for i in range(len(unique_massives)):
        test_massive = unique_massives[i]
        train_indices = []
        test_indices = dict_massives[test_massive]['indices']
        for j in range(len(unique_massives)):
            if j != i:
                train_indices.extend(dict_massives[unique_massives[j]]['indices'])
        return train_indices, test_indices

def combination_methode(x, y):
    pass  

class fold_management: 

    def __init__(self, shuffle=False, random_state=42, train_aprox_size=0.80):
        self.shuffle = shuffle
        self.seed = random_state
        self.rng = np.random.default_rng(self.seed)

    def split(self, x, y):
        massives_count = {}

        for index, name in enumerate(y['metadata'][:, 1]):
            if name not in massives_count:
                massives_count[name] = {'count': 0, 'indices': []}
            massives_count[name]['count'] += 1
            massives_count[name]['indices'].append(index)
            
        if np.unique(y['metadata'][:, 1]).size != 1:
            return KFold_methode(x, number_groups=5, seed=self.seed, shuffle=self.shuffle)
        
        return fold_massive_methode(massives_count)
