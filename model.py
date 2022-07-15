import pandas as pd
import numpy as np
import argparse
import logging
logging.basicConfig(level = logging.INFO)
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score
import mlflow
import mlflow.sklearn

if __name__ == "__main__":
    X = pd.read_csv("X.csv")
    y = pd.read_csv("y.csv")['Precio_leche']

    np.random.seed(0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with mlflow.start_run():

        pipe = Pipeline([('scale', StandardScaler()),
                        ('selector', SelectKBest(mutual_info_regression)),
                        ('poly', PolynomialFeatures()),
                        ('model', Ridge())])

        #Potential improvement: Make this an mlflow experiment
        K= [3, 4, 5, 6, 7, 10] 
        logging.info(f"Hyperparameter space for k in Ridge Regression: {','.join([str(k) for k in K])}")
        ALPHA= [1, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01] 
        logging.info(f"Hyperparameter space for alpha in Ridge Regression: {','.join([str(a) for a in ALPHA])}")
        POLY = [1, 2, 3, 5, 7] 
        logging.info(f"Hyperparameter space for polynomial defree features in Ridge Regression: {','.join([str(p) for p in POLY])}")

        grid = GridSearchCV(estimator = pipe,
                            param_grid = dict(selector__k=K,
                                              poly__degree=POLY,
                                              model__alpha=ALPHA),
                            cv = 3,
                            scoring = 'r2')

        grid.fit(X_train, y_train) 
        y_predicted = grid.predict(X_test)

        #log parameters
        logging.info("Best parameters")
        for k,v in grid.best_params_.items():
            logging.info(k + ":" + str(v))
            mlflow.log_param(k,v)

        # evaluar modelo
        rmse = mean_squared_error(y_test, y_predicted)
        r2 = r2_score(y_test, y_predicted)

        # log metrics
        logging.info(f"RMSE: {rmse}")
        mlflow.log_metric('RMSE', rmse)
        logging.info(f"R2: {r2}")
        mlflow.log_metric('R2', r2)

        # log the sklearn model
        mlflow.sklearn.log_model(grid,"sk-learn-ridge-regression")
        logging.info(f"Model saved in run {mlflow.active_run().info.run_uuid}")



