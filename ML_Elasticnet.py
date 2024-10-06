import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 

def gen_data(rows, cols):

    X = np.random.randn(rows, cols)

    t_coff = np.random.randn(cols)
    t_coff[2:5] = 0

    noise = np.random.randn(rows) * 0.5
    y = np.dot(X, t_coff) + noise

    df = pd.DataFrame(X, columns = [f'feature_{i+1}' for i in range(cols)])
    df['target'] = y
    
    x = df.drop("target", axis = 1)
    Y   = df["target"]
    
    return x, Y

def train_test_split(x, y, test_size):
    np.random.seed(10)
    
    rows = x.shape[0]
    test = int(rows * test_size)
    test_index = np.random.choice(rows, test, replace = False)
    
    dublicate = np.zeros(rows, dtype= bool)
    dublicate[test_index] = True
    
    X_train = x[~dublicate]
    X_test = x[dublicate]
    y_train = y[~dublicate].reset_index(drop = True)
    y_test = y[dublicate].reset_index(drop = True)
    
    return X_train,X_test,y_train,y_test


class ElasticNet: 
    def __init__(self, 
                 alpha=1.0, 
                 l_ratio=0.5,
                 learning_rate = 0.0005,
                 iterations = 1000):
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
                print(f"Iteration {i}, Loss: {loss}")
    
    def predict(self,X_test):
        return self.lin_reg_model(X_test)     

def standardize(X_train, X_test):
    mean = X_train.mean()
    std = X_train.std()
    X_train_std = (X_train - mean)/std
    X_test_std = (X_test - mean)/std
    return X_train_std, X_test_std

X, y = gen_data(1000, 10)

# df = pd.read_csv("")

# X = df.drop("target", axis=1)
# y = df[target]

X.shape, y.shape

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)

X_train.shape, X_test.shape, y_train.shape, y_test.shape

X_train_std, X_test_std = standardize(X_train, X_test)

X_train_std.shape, X_test_std.shape

model = ElasticNet(alpha = 0.01,
                   l_ratio = 0.2,
                   learning_rate =0.0005,
                   iterations= 10000)

model.fit(X_train_std.values, y_train.values)

y_pred =  model.predict(X_test_std.values)

mse = np.mean((y_test - y_pred)**2)
print(f"Mean Squared Error on Test Set : {mse}")

print("Learned Weights:", model.weights)
print("Learned Bias:", model.bias)

comparison_df = pd.DataFrame({
    'Actual Values': y_test.values,
    'Predicted Values': y_pred
})

comparison_df.head(20)

plt.figure(figsize=(10, 6))
plt.scatter(comparison_df.index, comparison_df['Actual Values'], color='blue', label='Actual Values', marker='o')
plt.scatter(comparison_df.index, comparison_df['Predicted Values'], color='red', label='Predicted Values', marker='x')
plt.plot(comparison_df.index, comparison_df['Actual Values'], color='blue', alpha=0.5)
plt.plot(comparison_df.index, comparison_df['Predicted Values'], color='red', alpha=0.5)

plt.title('Actual vs Predicted Values Comparison')
plt.xlabel('Index')
plt.ylabel('Values')
plt.axhline(0, color='black', linewidth=1.5, ls='--')
plt.axvline(0, color='black', linewidth=1.5, ls='--')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

