# Boston Housing Price Prediction — Model Comparison

A data mining exercise comparing seven regression modeling techniques to predict median home values using the classic Boston Housing dataset (MASS::Boston).

## Overview

This project benchmarks seven different regression approaches against each other on the same train/test split, using in-sample (ASE) and out-of-sample (MSPE) error to evaluate which method generalizes best to unseen data.

Best model: Random Forest — lowest out-of-sample error (MSPE = 8.17) with strong in-sample fit (ASE = 2.13), indicating it generalizes well without simply overfitting the training data.

## Dataset

Source: Boston dataset from the MASS package in R
Observations: 506 census tracts in the Boston area
Target variable: medv — median home value (in $1000s)
Predictors: 13 variables including crime rate (crim), average rooms per dwelling (rm), pupil-teacher ratio (ptratio), percent lower-status population (lstat), and others
Split: 80% train / 20% test (seed set to student M-number for reproducibility)

## Models Compared

Linear Model
Regression Tree
k-Nearest Neighbors
Random Forest
Boosting
GAM
Neural Network

## Results

MethodASE (in-sample)MSPE (out-of-sample)Linear Model23.1917.23Regression Tree13.7221.78k-NN (k=3)9.0911.53Random Forest2.138.17Boosting0.289.24GAM11.6812.62Neural Network (scaled X&Y)4.7019.00

Takeaway: Random Forest achieved the best out-of-sample performance overall. Boosting posted the lowest in-sample error (0.28), but its higher MSPE relative to Random Forest suggests some overfitting — a reminder that in-sample fit alone isn't a reliable indicator of real-world predictive performance.

## Tools Used

R with the following packages: MASS, tidyverse, ggplot2, rpart / rpart.plot, FNN, randomForest, gbm, mgcv, neuralnet

## How to Run

Open scripts/Boston_Weil_Jay.R in RStudio (or your R environment of choice).
Install any missing packages: install.packages(c("MASS","tidyverse","ggplot2","rpart","rpart.plot","FNN","randomForest","gbm","mgcv","neuralnet"))
Run the script top to bottom — it loads the data, performs EDA, fits all seven models, and compiles results into a summary table (boston_models_results).

## Author

Jay Weil

M.S. Business Analytics
B.S. Electrical Engineering
University of Cincinnati

## License

The Boston Housing dataset is distributed via the MASS R package.
