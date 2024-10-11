# ElasticNet Regression Implementation

## Project Members
The following members worked on this project:

- Atharv Patil (A20580674)
- Emile Mondon (A20600364)
- Tejaswini Viswanath (A20536544)
- Merlin Santano Simoes (A20531255)


### Requirements

- Python 3.7+
- Required Libraries: `numpy`, `pandas`, `pytest`, 'sys', 'argparse'

### File Structure

- `Models/ElasticNet.py`: Contains the `ElasticNetModel` class to perform ElasticNet regression.
- `Models/Data_Gen.py`: Contains `DataGenerator` and `ProfessorData` classes to generate datasets.
- `Test_Model.py`: Main script to train the ElasticNet model and evaluate performance metrics.
- `PyTest.py`: Script to run tests for model and data functions using PyTest.

---

### Setup

Clone the repository and ensure the required libraries are installed:
```bash
git clone https://github.com/your-repo-name/elastic-net-model
pip install -r requirements.txt
```
Also, make sure to change the path in `Test_Model.py` and `PyTest.py`  as needed:

```bash
import sys
sys.path.insert(0, '-- path of Root Directory -- ')
```
### How to Run Model_test.py

`Test_Model.py` accepts several arguments that allow you to configure data generation, Elastic Net parameters, and CSV file inputs. Below are instructions and examples for using each argument. 
- Note that use only one type of arguments at a time 

## Argument Options

### Generated Data Arguments:

- `--rows`: Number of rows/samples in the generated data.
- `--cols`: Number of columns/features in the generated data.
- `--noise`: Noise level in the generated data.
- `--seed`: Random seed for reproducibility.

### Example Command:

```bash
python Model_test.py --rows 100 --cols 10 --noise 0.1 --seed 42
```

### Professor Data Generation Arguments:

- `-N`: Number of samples.
- `-m`: Regression coefficients.
- `-b`: Offset (intercept).
- `-scale`: Scale of noise.
- `-rnge`: Range of values for features.
- `-random_seed`: Seed for reproducibility.

### Example Command:

```bash
python Model_test.py -N 100 -m 1.0 2.0 -b 0.5 -scale 0.1 -rnge 0.0 10.0 -random_seed 42
```

### CSV File Input Arguments:

- `--csv_file_path`: Path to the CSV file containing your dataset.
- `--target_column`: Name of the target column in the CSV file.

### Example Command:

```bash
python Model_test.py --csv_file_path "data/sample.csv" --target_column "Price"
```
### Elastic Net Model Arguments:

- `--alpha`: Regularization strength.
- `--penalty_ratio`: Ratio between L1 and L2 penalties.
- `--learning_rate`: Learning rate.
- `--iterations`: Number of iterations.

### Example Command:

```bash
python Model_test.py --alpha 0.01 --penalty_ratio 0.1 --learning_rate 0.001 --iterations 10000
```

### Test Set:
- `--test_size`: Fraction of data to be used for testing.

### Example Commands

### Generate Data and Train Model:

```bash
python Test_Model.py --rows 100 --cols 5 --noise 0.2 --alpha 0.01 --penalty_ratio 0.5 --learning_rate 0.001 --iterations 5000 --test_size 0.2
```
### Train Model on Data Generated from Professor Code

```bash
python Test_Model.py -N 100 -m 1.0 2.0 -b 0.5 -scale 0.1 -rnge 0.0 10.0 -random_seed 42 --alpha 0.01 --penalty_ratio 0.5 --learning_rate 0.001 --iterations 5000 --test_size 0.2
```

### Train Model on CSV Data:
```bash
python Test_Model.py --csv_file_path "data/sample.csv" --target_column "Price"
```
### Running Tests

To verify the functionality, you can run the test script `PyTest.py`, which includes unit tests for functions in the model pipeline. Make sure you are in the Test Directory before running the below command.

```bash
pytest PyTest.py
```

## Output
The script outputs the following evaluation metrics:

- **Mean Squared Error (MSE)**: Quantifies the prediction error by averaging the squares of the differences between the actual and predicted values. A lower MSE indicates better model performance.

- **Mean Absolute Error (MAE)**: Represents the average absolute error between the actual and predicted values. Like MSE, a lower MAE signifies a better fit of the model to the data.

- **R2 Score**: Measures the proportion of variance in the dependent variable that can be predicted from the independent variables. An R2 score closer to 1 indicates a better model fit.


## Q1a. What does the Model do 
The ElasticNet Linear Regression model implemented in the above ipynb file is a combination of L1 (Lasso) and L2 (Ridge) regularization. The Model is implemented to address colinear Data and Overfitting problems of linear regression. The model is highly useful if the dataset has high dimensionaliity where features may be corelated. It can improve prediction accuracy by balancing btoh L1 and L2 Penalties.

### Q1b. When to Use
- **Multicollinearity**: When the Feature variables are highly corerelated to the target variabole, This model (ElasticNet) will help slecting a subset of Features and managing their coefficients effectively.
- **High-Dimensional Data**: It Can be use with dataset which has less samples but the number of features are high, Our Model (ElasticNet) gives us a approch to avoide overfitting while still fitting the model.
- **Variable Selection**: It can be used when the goal is to find the most important featurs from a large dataset, the L1 Penalty in the ElasticNet Model helps in varialble selection.

## Q2. Testing the Model
To determine if the model is working correctly, I followed a systematic testing approach:

1. **Colinear Data Generation**: The model was tested using colinear data generated by the `gen_data` function, ensuring that multicollinearity was present.
2. **CSV Data Acceptance**: The model also accepts CSV data.
3. **Training and Testing**: The dataset was split into training and testing sets using the `train_test_split` method. The model was trained on the training set, and predictions were made on the separate test set.
4. **Performance Metrics**: The Mean Squared Error (MSE) was calculated on the test set to evaluate the accuracy of predictions. The goal was to achieve an MSE below 0.2.
5. **Monitoring Loss**: The loss (MSE), (MAE) and (R squared Score)  was printed at regular intervals during training to monitor convergence and ensure that the model was learning effectively.


## Q3. Tunable Parameters
The following parameters are exposed for tuning model performance:

- `alpha`: The regularization strength. Increasing this value increases the penalty on coefficients, which can reduce overfitting.
- `Penalty_ratio`: The mix ratio of L1 and L2 penalties. A value of 0.0 corresponds to L2 (Ridge) regression, while 1.0 corresponds to L1 (Lasso) regression. A value between 0 and 1 allows for a combination of both.
- `learning_rate`: The learning rate for gradient descent. This controls how much to change the model weights with respect to the loss gradient.
- `iterations`: The number of iterations for the gradient descent algorithm. Increasing this value may lead to better convergence but requires more computation time.

## Q4 Limitations and Future Work
Currently our Model may struggle with the following specific inputs:

- **Non-linear Data**: The model is inherently linear, which means it may not perform well on datasets that exhibit non-linear relationships. Given More time we could explore implementing polynomial features address this limitation.
- **Outliers**: The model's might not work well if significant outliers are present in the dataset, which can skew the loss calculation. Given more time more effective regression techniques could be exprimented to address this limitation.
- **Large Datasets**: The performance migh fall if we try when dealing with very large datasets due to computational cost. Given time experementing the use more efficient optimization algorithms like mini batch Gradient Descent could be give us positive results.

## Results
Here are the results from the model after training:

- **Mean Squared Error on Test Set**: 
    ```
    0.14815693888248077
    ```
- **Mean Absolute Error on Test Set**: 
    ```
    0.31090431686292846
    ```
- **R2 Score on Test Set**: 
    ```
    0.9808019625757901
    ```

### Visualization of Results
The following plot illustrates the comparison between actual and predicted values compared with SkLearn Model:

![Actual vs Predicted Values Comparison]([https://github.com/AtharvPat/ML-Project_-1-/blob/main/Results/output.png](https://github.com/AtharvPat/ML-Project_-1-/blob/main/PROJECT/Notebooks/Visilizuations/genData%20visulaization.png))

