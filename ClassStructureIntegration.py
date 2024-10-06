import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class ElasticNetModel():
    def __init__(self, alpha=1.0, l_ratio=0.5, learning_rate=0.0005, iterations=1000):
        """
        Initialize the ElasticNetModel with custom parameters.
        """
        self.model = ElasticNet(alpha=alpha, l_ratio=l_ratio, learning_rate=learning_rate, iterations=iterations)
        self.results = None

    def fit(self, X_train, y_train):
        """
        Fit the ElasticNet model using training data.
        """
        self.model.fit(X_train, y_train)
        self.results = ElasticNetModelResults(self.model)
        return self.results

    def gen_data(self, rows, cols):
        """
        Generate synthetic data for testing.
        """
        X = np.random.randn(rows, cols)
        t_coff = np.random.randn(cols)
        t_coff[2:5] = 0
        noise = np.random.randn(rows) * 0.5
        y = np.dot(X, t_coff) + noise

        df = pd.DataFrame(X, columns=[f'feature_{i+1}' for i in range(cols)])
        df['target'] = y

        X = df.drop("target", axis=1)
        y = df["target"]

        return X, y

    def train_test_split(self, X, y, test_size=0.2):
        """
        Manually split the data into training and test sets.
        """
        np.random.seed(10)
        rows = X.shape[0]
        test = int(rows * test_size)
        test_index = np.random.choice(rows, test, replace=False)

        duplicate = np.zeros(rows, dtype=bool)
        duplicate[test_index] = True

        X_train = X[~duplicate]
        X_test = X[duplicate]
        y_train = y[~duplicate].reset_index(drop=True)
        y_test = y[duplicate].reset_index(drop=True)

        return X_train, X_test, y_train, y_test

    def standardize(self, X_train, X_test):
        """
        Standardize the training and test data.
        """
        mean = X_train.mean()
        std = X_train.std()
        X_train_std = (X_train - mean) / std
        X_test_std = (X_test - mean) / std
        return X_train_std, X_test_std


class ElasticNetModelResults():
    def __init__(self, model):
        """
        Store the trained ElasticNet model.
        """
        self.model = model

    def predict(self, X_test):
        """
        Predict the target values for the provided input data X_test.
        """
        return self.model.predict(X_test)

    def print_comparison(self, y_test, y_pred):
        """
        Print the actual vs predicted values.
        """
        comparison_df = pd.DataFrame({
            'Actual Values': y_test.values,
            'Predicted Values': y_pred
        })
        print(comparison_df.head(20))  # Print the first 20 rows of actual vs predicted values

    def plot_comparison(self, y_test, y_pred):
        """
        Plot the actual vs predicted values.
        """
        comparison_df = pd.DataFrame({
            'Actual Values': y_test.values,
            'Predicted Values': y_pred
        })

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


class ElasticNet:
    def __init__(self, alpha=1.0, l_ratio=0.5, learning_rate=0.0005, iterations=1000):
        self.alpha = alpha
        self.l_ratio = l_ratio
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = 0

    def l1_penalty(self):
        return self.l_ratio * np.sign(self.weights)

    def l2_penalty(self):
        return (1 - self.l_ratio) * self.weights

    def lin_reg_model(self, X_test):
        return np.dot(X_test, self.weights) + self.bias

    def fit(self, X_train, y_train):
        rows, cols = X_train.shape
        self.weights = np.zeros(cols)

        for i in range(self.iterations):
            y_pred = self.lin_reg_model(X_train)
            residuals = y_pred - y_train

            gradients_w = (1 / rows) * np.dot(X_train.T, residuals)
            gradients_b = (1 / rows) * np.sum(residuals)

            self.weights -= self.learning_rate * (gradients_w + self.alpha * (self.l1_penalty() + self.l2_penalty()))
            self.bias -= self.learning_rate * gradients_b

            if i % 1000 == 0:
                loss = (1 / rows) * np.sum((y_train - y_pred) ** 2)
                print(f"Iteration {i}, Loss: {loss}")

    def predict(self, X_test):
        return self.lin_reg_model(X_test)


# Example usage:

model = ElasticNetModel(alpha=0.01, l_ratio=0.2, learning_rate=0.0005, iterations=10000)

X, y = model.gen_data(1000, 10)

X_train, X_test, y_train, y_test = model.train_test_split(X, y, test_size=0.2)
X_train_std, X_test_std = model.standardize(X_train, X_test)

results = model.fit(X_train_std.values, y_train.values)
y_pred = results.predict(X_test_std.values)

# Print actual vs predicted values
results.print_comparison(y_test, y_pred)

# Calculate MSE
mse = np.mean((y_test - y_pred) ** 2)
print(f"Mean Squared Error on Test Set: {mse}")

# Plot results
results.plot_comparison(y_test, y_pred)
