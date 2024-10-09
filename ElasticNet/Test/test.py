import pandas as pd 
import numpy as np
from models.ElasticNet import ElasticNetModel

def gen_data(rows, cols):
    np.random.seed(10)
    X = np.random.randn(rows, cols)

    t_coff = np.random.randn(cols)
    t_coff[2:5] = 0

    noise = np.random.randn(rows) * 0.4
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

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)

# X_train.shape, X_test.shape, y_train.shape, y_test.shape

X_train_std, X_test_std = standardize(X_train, X_test)

# X_train_std.shape, X_test_std.shape

model = ElasticNetModel(alpha = 0.01,
                   l_ratio = 0.1,
                   learning_rate =0.001,
                   iterations= 10000)

model.fit(X_train_std.values, y_train.values)

y_pred =  model.predict(X_test_std.values)

# print("Learned Weights:", model.weights)
# print("Learned Bias:", model.bias)

comparison_df = pd.DataFrame({
    'Actual Values': y_test.values,
    'Predicted Values': y_pred
})

comparison_df["difference"] = comparison_df['Actual Values'] - comparison_df['Predicted Values']

mse = np.square(comparison_df["difference"]).mean()

print(f"Mean Squared Error on Test Set : {mse}")

mae = np.abs(comparison_df['difference']).mean()

print(f"Mean Absolute Error on Test Set : {mae}")

r2 = model.r2_score(y_test.values, y_pred)
print(f"R² Score on Test Set : {r2}")