import pandas as pd 
import numpy as np

class ElasticNetModel:      
    def __init__(self, 
                 alpha=0.01, 
                 l_ratio=0.1,
                 learning_rate = 0.001,
                 iterations = 150000):
        self.alpha = alpha
        self.l_ratio = l_ratio
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = 0
        
    
    def l1_penalty(self):
        l1_penalty  = self.l_ratio * np.sign(self.weights)
        return l1_penalty
    
    def l2_penalty(self):
        l2_penalty  = (1 - self.l_ratio) * self.weights
        return l2_penalty

    def lin_reg_model(self, X_test):
        return np.dot(X_test, self.weights) + self.bias
            
    def fit(self, X_train, y_train):
        # X = X_train.to_numpy()
        # y = y_train.to_numpy()
        rows, cols = X_train.shape
        self.weights = np.zeros(cols)
        
        # self.intercept = 0
        
        # feature_conbination_X = X_train.T.dot(X_train)
        # feature_conbination_y = X_train.T.dot(y_train)
        
        for i in range(self.iterations):
            y_pred = self.lin_reg_model(X_train)
            residuals = y_pred- y_train
            
            gradients_w = (1 / rows) * np.dot(X_train.T, residuals)
            gradients_b = (1 / rows) * np.sum(residuals)
            
            self.weights -= self.learning_rate *(gradients_w + self.alpha *(self.l1_penalty()+ self.l2_penalty()))
            self.bias -= self.learning_rate *gradients_b
            
            if i % 1000 == 0:
                loss = (1/rows) *np.sum((y_train - y_pred)** 2) 
                # print(f"Iteration {i}, Loss: {loss}")
    
    def predict(self,X_test):
        return self.lin_reg_model(X_test)     