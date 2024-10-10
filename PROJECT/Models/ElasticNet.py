import numpy as np
import pandas as pd

class ElasticNetModel: 
    def __init__(self,
                 alpha =0.01,
                 penalty_ratio = 0.1,
                 learning_rate = 0.001,
                 iterations = 150000):
        
        self.alpha = alpha
        self.penalty_ratio = penalty_ratio
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = 0 
        
    def linear_reg_model(self, x_test): 
        return np.dot(x_test, self.weights) + self.bias
    
    def l1_penalty(self):
        return self.penalty_ratio * np.sign(self.weights)
    
    def l2_penalty(self):
        return (1 - self.penalty_ratio) * self.weights
    

    
    def fit(self, X_train, y_train):
        
        if isinstance(X_train, (pd.DataFrame, pd.Series)):
            X_train = X_train.to_numpy()
        if isinstance(y_train, (pd.DataFrame, pd.Series)):
            y_train = y_train.to_numpy()
        
        rows, cols = X_train.shape
        # self.weights = np.zeros(cols)   
        
        # if y_train.ndim > 1:
        #     y_train = y_train.flatten()
        
        self.weights = np.zeros(cols)
        
        for i in range(self.iterations):
            y_pred = self.linear_reg_model(X_train)
            residuals = y_pred - y_train
            
            # print(residuals)
            
            gradients_w = (1 / rows) * np.dot(X_train.T, residuals)
            gradients_b = (1 / rows) * np.sum(residuals)
            
            # print(gradients_w,gradients_b)
            
            self.weights -= self.learning_rate * (gradients_w + self.alpha * (self.l1_penalty() + self.l2_penalty()))
            self.bias -= self.learning_rate * gradients_b
            
            if i % 1000 == 0:
                loss = (1 / rows) * np.sum((y_train - y_pred) ** 2) 
                # print(f"Iteration {i}, Loss: {loss}")
  
    def predict(self, X_test):
        if isinstance(X_test, (pd.DataFrame, pd.Series)):
            X_test = X_test.to_numpy()
        return self.linear_reg_model(X_test)
                
class ElasticNetModelResults():
    def __init__(self,
                #   model, 
                #   X_test,
                  y_test,
                  y_pred):
        # self.model = model
        # self.X_test = X_test
        self.y_test = y_test
        self.y_pred = y_pred
         
         
    def r2_score(self, y_test, y_pred):
        square_total = np.sum((y_test - np.mean(y_test))**2)
        square_remainder = np.sum((y_test - y_pred)**2)
        
        return 1-(square_remainder/square_total)
            
        