import numpy as np
import pandas as pd
import sys 
import argparse

sys.path.insert(1, 'E:/IIT CHICAGO/SEMESTER_1/MACHINE LEARNING/PROJECT')

from  Models.ElasticNet import ElasticNetModel
from  Models.ElasticNet import ElasticNetModelResults
from  Models.Data_Gen import DataGenerator
from  Models.Data_Gen import ProfessorData

def train_test_split(x, y, test_size): 
    np.random.seed(10)
    
    rows = x.shape[0]
    test = int(rows * test_size)
    test_index = np.random.choice(rows,test, replace= False)
    
    dublicate = np.zeros(rows, dtype=bool)
    dublicate[test_index] = True

    if isinstance(x, pd.DataFrame):
        X_train = x[~dublicate].reset_index(drop=True)
        X_test = x[dublicate].reset_index(drop=True)
    else:
        X_train = x[~dublicate]
        X_test = x[dublicate]
    
    if isinstance(y, pd.Series):
        y_train = y[~dublicate].reset_index(drop=True)
        y_test = y[dublicate].reset_index(drop=True)
    else:
        y_train = y[~dublicate]
        y_test = y[dublicate]
    
    return X_train, X_test, y_train, y_test

def standardize(X_train, X_test): 
    mean = X_train.mean()
    standard_deviation= X_train.std()
    X_train_std = (X_train-mean)/standard_deviation
    X_test_std  = (X_test-mean)/standard_deviation  
    return X_train_std, X_test_std

def parse_arguments():
    parser = argparse.ArgumentParser(description= " Train an Elastic Net Model on Generated Data")
    
    parser.add_argument('--rows', type=int,  help ="Number of Rows/Samples in Generated Data")
    parser.add_argument('--cols', type=int,  help ="Number of Columns/features in Generated Data")
    parser.add_argument('--noise', type=float, help ="Noise Scale in Generated Data")
    parser.add_argument('--seed', type=int,  help ="Random Seed for Data Generation")
    
    # Argument Parser for Professor data 
    parser.add_argument("-N", type = int, help="Number of samples.")
    parser.add_argument("-m", nargs='+', type = float, help="Expected regression coefficients")
    parser.add_argument("-b", type= float, help="Offset")
    parser.add_argument("-scale", type=float, help="Scale of noise")
    parser.add_argument("-rnge", nargs=2, type=float,  help="Range of Xs")
    parser.add_argument("-random_seed", type=int, help="A seed to control randomness")
   
    parser.add_argument('--test_size', type=float, default=0.2, help ="Size of Test Set")
    
    parser.add_argument('--alpha', type=float, default=0.01, help ="Regularization strength for the ElasticNet Model ")
    parser.add_argument('--penalty_ratio', type=float, default=0.1, help ="Ratio between lasso(L1) and Ridge(L2) penalties")
    parser.add_argument('--learning_rate', type=float, default=0.001, help ="Model training Learning Rate")
    parser.add_argument('--iterations', type=int, default=10000, help ="Iterations for Training the model")
    
    parser.add_argument('--csv_file_path', type=str ,help ="Csv file path")
    parser.add_argument('--target_column', type=str, help ="Name of the target column in the csv file")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    
    if args.csv_file_path and args.target_column:
        df = pd.read_csv(args.csv_file_path)
        print("Columns in DataFrame:", df.columns.tolist())

        if args.target_column in df.columns:
            X = df.drop(args.target_column, axis=1)
            y = df[args.target_column]
        else:
            raise KeyError(f"Column '{args.target_column}' not found in DataFrame.")

        
    elif args.rows and args.cols and args.noise:
        data_gen = DataGenerator(rows=args.rows, cols=args.cols, noise=args.noise, seed=args.seed)
        X, y = data_gen.gen_data()
        
    else: 
        data_gen = ProfessorData(rnge=args.rnge, scale=args.scale, m=args.m, b=args.b, N=args.N, random_seed=args.random_seed)
        X, y = data_gen.linear_data_generator()

    
    # print(X.shape)
    # print(y.shape)
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= args.test_size)
    
    X_train_std, X_test_std = standardize(X_train, X_test)
    
    model = ElasticNetModel(alpha = args.alpha,
                            penalty_ratio = args.penalty_ratio,
                            learning_rate = args.learning_rate,
                            iterations = args.iterations)
    
    X_train_std_np = X_train_std.values if isinstance(X_train_std, pd.DataFrame) else X_train_std
    y_train_np = y_train.values if isinstance(y_train, pd.Series) else y_train
    X_test_std_np = X_test_std.values if isinstance(X_test_std, pd.DataFrame) else X_test_std
    y_test_np = y_test.values if isinstance(y_test, pd.Series) else y_test
    
    # print(X_train_std_np.shape)
    # print(y_train.shape)
    model.fit(X_train_std_np,y_train_np)
    y_pred = model.predict(X_test_std_np) 
    # print(y_pred)
    
    comparison_df = pd.DataFrame({
        'Actual Values': y_test_np,
        'Predicted Values': y_pred
    })

    comparison_df["Difference"] = comparison_df['Actual Values'] - comparison_df['Predicted Values']


    mse = np.square(comparison_df["Difference"]).mean()
    print(f"Mean Squared Error on Test Set: {mse}")

    mae = np.abs(comparison_df['Difference']).mean()
    print(f"Mean Absolute Error on Test Set: {mae}")
    
    results = ElasticNetModelResults(y_test=y_test, y_pred = y_pred)
    r2 = results.r2_score(y_test_np, y_pred)
    print(f"R2 Score on Test Set: {r2}")
    
    

    

        
    

    
    
    
    

    