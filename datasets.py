
import numpy as np
import scipy.stats as sps
import pickle

class DataSet:
    
    def __init__(self, count=100, nClasses=10, L=20, alpha_high=1, beta_high=10, lna=4, lnb=1, random_seed=42):
        np.random.seed(seed=random_seed)
        m = sps.beta.rvs(alpha_high, beta_high, size=count)
        s = np.exp(sps.norm(lna, lnb).rvs(size=count))
        
        self.L = L
        self.nClasses = nClasses
        self.count = count
        self.alpha0 = s * m
        self.beta0 = s * (1 - m)
        self.p = sps.beta.rvs(self.alpha0, self.beta0, size=(nClasses, count)).T
        self.train_data = sps.binom.rvs(n=L, p=self.p)
        self.val_data = sps.binom.rvs(n=L, p=self.p)
        #self.test_data = sps.binom.rvs(n=L, p=self.p)
        self.ideal = sps.binom.rvs(n=10 ** 6, p=self.p)
        
def load_data_set(filename):
    with open(filename, 'rb') as input_:
        return pickle.load(input_)
        
def save_data_set(filename, dataset):
    with open(filename, 'wb') as output:
        pickle.dump(dataset, output, pickle.HIGHEST_PROTOCOL)
