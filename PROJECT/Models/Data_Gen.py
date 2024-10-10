import numpy as np
import pandas as pd 

class DataGenerator:
    def __init__(self, rows, cols, noise =0.4, seed=10, range =[-10,10], noise_scale = 0.4, size =(100,10), random_seed = 8675309 ):
        self.rows = rows
        self.cols = cols
        self.noise = noise
        self.seed = seed
        self.noise_scale = noise_scale
        self.range = range
        self.size = size
        self.random_seed = random_seed
        
    def gen_data(self):
        np.random.seed(10)
        X = np.random.randn(self.rows, self.cols)

        t_coff = np.random.randn(self.cols)
        t_coff[2:5] = 0

        noise = np.random.randn(self.rows) * 0.4
        y = np.dot(X, t_coff) + noise

        df = pd.DataFrame(X, columns = [f'feature_{i+1}' for i in range(self.cols)])
        df['target'] = y
        
        x = df.drop("target", axis = 1)
        Y   = df["target"]
        
        return x, Y
    
class ProfessorData:    
    def __init__(self,m,N,b,rnge =[-10,10], scale = 1, random_seed = 8675309 ):
        self.scale = scale
        self.rnge = rnge
        self.m = m
        self.N = N
        self.b = b
        self.random_seed = random_seed    
        
    ## Professor Code to Generate Collinear Data 
    def linear_data_generator(self):
        rng = np.random.default_rng(seed=self.random_seed)
        sample = rng.uniform(low=self.rnge[0], high=self.rnge[1], size=(self.N, len(self.m)))
        m_reshaped = np.array(self.m).reshape(-1,1)
        ys = np.dot(sample, m_reshaped) + self.b
        noise = rng.normal(loc=0., scale=self.scale, size=ys.shape)
        return sample, (ys + noise).flatten()
