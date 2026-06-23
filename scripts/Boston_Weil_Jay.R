# Individual Boston Housing Case Study
# BANA 7047-001
# Weil, Jay
# M14593230

# Libraries ---------------------------------------------------------------
# load in packages
library(MASS)          # Boston housing data set
library(tidyverse)     # quality of life package
library(ggplot2)       # plotting
library(rpart)         # regression tree
library(rpart.plot)    # tree plot
library(FNN)           # k-NN regression
library(randomForest)  # random forest
library(gbm)           # boosting
library(mgcv)          # GAM
library(neuralnet)     # neural network
set_theme(theme_bw())

# Load data ---------------------------------------------------------------

# data
data("Boston")
boston_df <- Boston
view(boston_df)

# Set seed and create 80/20 train-test split ----------------------------------------------------------------

# use M# to set seed
set.seed(14593230)

# 80/20 train-test split
n <- nrow(boston_df)
train_index <- sample(1:n, size = 0.8 * n)
train <- boston_df[train_index, ]
test <- boston_df[-train_index, ]

# Create function for MSE calculation in each model ----------------
# MSE helper function
mse <- function(actual, pred) {
  mean((actual - pred)^2)
  }

#  Create data frame for result storing -----------------------------------

# result storing data frame
boston_models_results <- data.frame(Method = character(), ASE = numeric(), MSPE = numeric(), stringsAsFactors = FALSE)

# Exploratory data analysis -----------------------------------------------

# check structure of data set
str(boston_df)
summary(boston_df)

# check for missing values in data set
colSums(is.na(boston_df))

# correlation of all predictors with target variable (medv)
cor_w_target <- cor(boston_df)[, "medv"]
sort(cor_w_target, decreasing = TRUE)

# histogram visualization for distribution of target variable (medv)
hist(boston_df$medv, main = "Distribution of MEDV", xlab = "MEDV")

# scatter plot matrix for variable relationship visualization
pairs(boston_df[, c("medv","rm","lstat","crim")])

# scatter plot with fitted linear trend to examine relationship between lstat and medv
ggplot(boston_df, aes(x = lstat, y = medv)) +
  geom_point() +
  geom_smooth(method = "lm") +
  labs(title = "MEDV vs LSTAT")

# Model 1: Linear Model ---------------------------------------------------

# fit linear model with medv as response variable
lm_fit <- lm(medv ~ ., data = train)
lm_fit

# get predictions on train and test sets
lm_predic_train <- predict(lm_fit, newdata = train)
lm_predic_train
lm_predic_test <- predict(lm_fit, newdata = test)
lm_predic_test

# calculate training and test error
lm_ase <- mse(train$medv, lm_predic_train)
lm_ase

lm_mspe <- mse(test$medv, lm_predic_test)
lm_mspe

# store results in table format
boston_models_results <- rbind(boston_models_results, data.frame(
  Method = "Linear Model", ASE = lm_ase, MSPE = lm_mspe, stringsAsFactors = FALSE))

# Model 2: Regression Tree with Pruning ------------------------------------------------

# fit regression tree model with medv as response variable
reg_tree_fit <- rpart(medv ~ ., data = train, method = "anova")
reg_tree_fit

# visualize the regression tree
rpart.plot(reg_tree_fit, main = "Regression Tree")

# start pruning by choosing complexity parameter with min error
complexity_param <- reg_tree_fit$cptable[which.min(reg_tree_fit$cptable[, "xerror"]), "CP"]
complexity_param

# prune the tree
reg_tree_prune <- prune(reg_tree_fit, cp = complexity_param)
reg_tree_prune

# visualize the pruned regression tree
rpart.plot(reg_tree_prune, main = "Pruned Regression Tree")

# get predictions on train and test sets
reg_tree_predic_train <- predict(reg_tree_prune, newdata = train)
reg_tree_predic_train
reg_tree_predic_test <- predict(reg_tree_prune, newdata = test)
reg_tree_predic_test

# calculate training and test error
reg_tree_ase <- mse(train$medv, reg_tree_predic_train)
reg_tree_ase

reg_tree_mspe <- mse(test$medv, reg_tree_predic_test)
reg_tree_mspe

# store results in table format
boston_models_results <- rbind(boston_models_results, data.frame(
  Method = "Regression Tree", ASE = reg_tree_ase, MSPE = reg_tree_mspe, stringsAsFactors = FALSE))

# Model 3: k-NN with optimal k = (scaled x) -------------------------------

# separate response and predictors
x_train <- subset(train, select = -medv)
x_test <- subset(test, select = -medv)

y_train <- train$medv
y_test <- test$medv

# scale X predictors using the training data
scale_x_train_mean <- apply(x_train, 2, mean)
scale_x_train_sd <- apply(x_train, 2, sd)

scale_x_train <- scale(x_train, center = scale_x_train_mean, scale = scale_x_train_sd)
scale_x_test <- scale(x_test, center = scale_x_train_mean, scale = scale_x_train_sd)

# choose optimal k value with test MSPE
k_values <- 1:20
knn_mspe_values <- numeric(length(k_values))

for (i in seq_along(k_values)) { 
  k <- k_values[i]
  
  knn_pred_test <- knn.reg(
    train = scale_x_train,
    test = scale_x_test,
    y = y_train,
    k = k
  )$pred
  knn_mspe_values[i] <- mse(y_test, knn_pred_test)
  }

# find optimal k
optimal_k <- k_values[which.min(knn_mspe_values)]
optimal_k

# training and test predictions using optimal k
knn_pred_train <- knn.reg(
  train = scale_x_train,
  test = scale_x_train,
  y = y_train,
  k = optimal_k
)$pred
knn_pred_train

knn_pred_test <- knn.reg(
  train = scale_x_train,
  test = scale_x_test,
  y = y_train,
  k = optimal_k
)$pred
knn_pred_test

# calculate training and test error
knn_ase <- mse(y_train, knn_pred_train)
knn_mspe <- mse(y_test, knn_pred_test)

knn_ase
knn_mspe

# store results in table format
boston_models_results <- rbind(boston_models_results, data.frame(
  Method = paste0("k-NN (k=", optimal_k, ")"), ASE = knn_ase, MSPE = knn_mspe, stringsAsFactors = FALSE))

# Model 4: Random Forests  ------------------------------------------------

# fit random forest model 
ran_for_fit <- randomForest(medv ~., data = train)
ran_for_fit

# training and test predictions
ran_for_predic_train <- predict(ran_for_fit, newdata = train)
ran_for_predic_train

ran_for_predic_test <- predict(ran_for_fit, newdata = test)
ran_for_predic_test

# calculate training and test error
ran_for_ase <- mse(train$medv, ran_for_predic_train)
ran_for_ase  

ran_for_mspe <- mse(test$medv, ran_for_predic_test)
ran_for_mspe

# store results in table format
boston_models_results <- rbind(boston_models_results, data.frame(
  Method = "Random Forest", ASE = ran_for_ase, MSPE = ran_for_mspe, stringsAsFactors = FALSE))
                   
# Model 5: Boosting  ------------------------------------------------------

# fit boosting model
boosting_fit <- gbm(medv~ ., data = train, distribution = "gaussian", n.trees = 2500, interaction.depth = 3, shrinkage = 0.05, cv.folds = 5,
                    n.minobsinnode = 10, verbose = FALSE)
boosting_fit

# find number of trees to use in cross validation
opt_num_trees <- gbm.perf(boosting_fit, method = "cv", plot.it = FALSE)
opt_num_trees

# training and test predictions
boosting_predic_train <- predict(boosting_fit, newdata = train, n.trees = opt_num_trees)
boosting_predic_train

boosting_predic_test <- predict(boosting_fit, newdata = test, n.trees = opt_num_trees)
boosting_predic_test

# calculate training and test error
boosting_ase <- mse(train$medv, boosting_predic_train)
boosting_ase

boosting_mspe <- mse(test$medv, boosting_predic_test)
boosting_mspe

# store results in table format
boston_models_results <- rbind(boston_models_results, data.frame(
  Method = "Boosting", ASE = boosting_ase, MSPE = boosting_mspe, stringsAsFactors = FALSE))
 
# Model 6: GAM ------------------------------------------------------------

# fit GAM model with auto variable selection
gam_fit <- gam(medv ~ s(crim) + zn + indus + chas + nox + s(rm) + age + s(dis) + rad + tax + ptratio + black + s(lstat), 
               data = train, method = "REML", select = TRUE)
gam_fit

# training and test predictions
gam_predic_train <- predict(gam_fit, newdata = train)
gam_predic_train

gam_predic_test <- predict(gam_fit, newdata = test)
gam_predic_test

# calculate training and test error
gam_ase <- mse(train$medv, gam_predic_train)
gam_ase

gam_mspe <- mse(test$medv, gam_predic_test)
gam_mspe

# store results in table format
boston_models_results <- rbind(boston_models_results, data.frame(
  Method = "GAM", ASE = gam_ase, MSPE = gam_mspe, stringsAsFactors = FALSE))

# Model 7: Neural Network (scaled X&Y) ------------------------------------

# scale response and predictors
nnet_train <- train
nnet_test <- test

# scale columns using training statistics
train_mean_nnet <- apply(nnet_train, 2, mean)
train_sd_nnet <- apply(nnet_train, 2, sd)

scale_nnet_train <- as.data.frame(scale(nnet_train, center = train_mean_nnet, scale = train_sd_nnet))
scale_nnet_test <- as.data.frame(scale(nnet_test, center = train_mean_nnet, scale = train_sd_nnet))

# fit neural network which has 1 hidden layer and 5 nodes
nnet_fit <- neuralnet::neuralnet(medv ~ ., data = scale_nnet_train, hidden = c(5), linear.output = TRUE)
nnet_fit

# training and test predictions (scaled)
nnet_predic_scaled_train <- neuralnet::compute(nnet_fit, scale_nnet_train[, -which(names(scale_nnet_train) == "medv")])$net.result
nnet_predic_scaled_test <- neuralnet::compute(nnet_fit, scale_nnet_test[, -which(names(scale_nnet_test) == "medv")])$net.result

# un-scale training and test predictions back to original scale
nnet_predic_unscaled_train <- as.numeric(nnet_predic_scaled_train) * train_sd_nnet["medv"] + train_mean_nnet["medv"]
nnet_predic_unscaled_test <- as.numeric(nnet_predic_scaled_test) * train_sd_nnet["medv"] + train_mean_nnet["medv"]

# calculate training and test error
nnet_ase <- mse(train$medv, nnet_predic_unscaled_train)
nnet_ase

nnet_mspe <- mse(test$medv, nnet_predic_unscaled_test)
nnet_mspe

# store results in table format
boston_models_results <- rbind(boston_models_results, data.frame(
  Method = "Neural Networks (scaled X&Y)", ASE = nnet_ase, MSPE = nnet_mspe, stringsAsFactors = FALSE))


# Get results to only be to two decimal places ------------------------------------

# round results to two decimals
boston_models_results$ASE <- round(boston_models_results$ASE, 2)
boston_models_results$MSPE <- round(boston_models_results$MSPE, 2)















