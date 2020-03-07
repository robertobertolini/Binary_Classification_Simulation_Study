# Simulation Study Code for:

# No Measurement Error
# 2n
# 25
# Missing Completely at Random
# XGB

# Last Modified: 3/7/2020

Sys.setenv(JAVA_HOME='')
library(earth)
library(randomForest)
library(DMwR)
library(caret)
library(caretEnsemble)
library(pROC)
library(glmnet)
library(plotROC)
library(tictoc)
library(mice)
library(gtools)
library(data.table)
library(readxl)
library(openxlsx)

set.seed(6) # Random seed used for all 500 iterations

auc_list <- c() # List to store the AUC values
mod <- c() # List to store the tuning parameters at each iteration for the method

# Name of the file that will output the AUC values. Its name consists
# of the four data mining properties and the method from the caret package

of="NoError_2n_25_MCAR_XGB.csv"

# Th execution time will also be recorded

tic("timer")

# 500 iterations of this program will be run

for (i in 1:500){
  
  n = 1500 # Size of the training + testing corpus
  
  # Generate 12 predictors from a standard normal distribution with mean 0 & var 1
  
  x1 = rnorm(n,mean = 0,sd = 1)         
  x2 = rnorm(n,mean = 0,sd = 1) 
  x3 = rnorm(n,mean = 0,sd = 1)         
  x4 = rnorm(n,mean = 0,sd = 1) 
  x5 = rnorm(n,mean = 0,sd = 1)          
  x6 = rnorm(n,mean = 0,sd = 1) 
  x7 = rnorm(n,mean = 0,sd = 1)         
  x8 = rnorm(n,mean = 0,sd = 1) 
  x9 = rnorm(n,mean = 0,sd = 1)         
  x10 = rnorm(n,mean = 0,sd = 1) 
  x11 = rnorm(n,mean = 0,sd = 1)         
  x12 = rnorm(n,mean = 0,sd = 1) 
  
  # Logistic Equation
  z = -3 + .75*x1 + .75*x2 + .75*x3 + .75*x4 + .75*x5 + .75*x6+rnorm(1,0,0.0001)       # linear combination with a bias
  pr = 1/(1+exp(z)) # Inverted logit function for the majority class
  y = rbinom(n,1,pr) # Bernoulli response variable
  
  # Create a dataframe with the independent variables and response variable
  data_mat <- as.data.frame(cbind(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,y))
  
  # Class imbalance: 25% minority class and 75% majority outcome
  test_fail <- data_mat[ sample( which(data_mat$y==0), 125), ]
  test_pass <- data_mat[ sample( which(data_mat$y==1), 375), ]
  
  testing_data <- rbind(test_fail,test_pass)
  
  # Divide the data into training and testing sets
  training_data <- subset(data_mat, !(rownames(data_mat) %in% rownames(testing_data)))
  train_dep <- training_data$y
  testing_data <- rbind(test_fail,test_pass)
  training_data <- subset(data_mat, !(rownames(data_mat) %in% rownames(testing_data)))
  train_dep <- training_data$y

  # Data Amputation: Missing Completely at Random
  data_mat_final <- ampute(data = training_data[,1:ncol(training_data)-1], prop = 0.6, mech = 'MCAR')$amp
  
  # After applying amputation, we reorganize the corpus
  data_mat_final$index <- as.numeric(row.names(data_mat_final))
  data_mat_final <- data_mat_final[order(data_mat_final$index), ]
  data_mat_final <- subset(data_mat_final, select = -c(index))
  data_original <- data_mat_final
  eve_data <- cbind(data_original,train_dep)
  names(eve_data)[names(eve_data) == 'train_dep'] <- 'y'
  training_data <- eve_data

  # Apply MICE to fill in the missing entries of the training data
  mice_training <- mice(training_data,m=1,maxit=50,meth='pmm',seed=500)
  training_data <- complete(mice_training,1)
  
  # Convert the dependent variable to pass and fail
  training_data$y[training_data$y == "0"] <- "F"
  training_data$y[training_data$y == "1"] <- "P"
  testing_data$y[testing_data$y == "0"] <- "F"
  testing_data$y[testing_data$y == "1"] <- "P"
  
  # Convert the dependent variable to a factor
  training_data$y <- factor(training_data$y)
  testing_data$y <- factor(testing_data$y)
  
  # Apply SMOTE to the training data
  training_data <- SMOTE(y ~ ., data = training_data)
  
  # 10-fold cross-validation will be applied to the training data
  ctrl = trainControl(method = "repeatedcv", repeats  = 1, classProbs = T, savePredictions = T, summaryFunction = twoClassSummary)
  mymethods = c("xgbLinear") # Data mining method
  out = caretList(y~., data = training_data, methodList = mymethods, trControl = ctrl, tuneLength = 6) # Train the model
  
  # Apply the model to the testing data and calculate the AUC on the testing corpus
  model_preds_tst = lapply(out, predict, newdata = testing_data[, 1:(dim(testing_data)[2] - 1)], type = "prob")
  model_preds_tst = lapply(model_preds_tst, function(x)x[,"F"])
  model_preds_tst = as.data.frame(model_preds_tst)[,-4]
  auc_test = caTools::colAUC(model_preds_tst, testing_data$y == "F", plotROC = T)
  auc_list[i] <- auc_test
  
  # Store the tuning parameters for each iteration in a csv spreadsheet
  
  if (i > 1){
    mod <- rbind(mod,out$xgbLinear$bestTune)
  }else{
    mod <- data.frame(out$xgbLinear$bestTune)
  }
  
  print(i) 
  rm(data_mat,testing_data)
  
}

write.csv(mod,'NoError_2n_25_MCAR_XGB_OUT.csv') # CSV file with parameters

print('')
toc(log=TRUE) # Record the execution time
boxplot(auc_list) # Generate a boxplot of the AUC values

write.csv(auc_list,file=paste('AUC',paste(mymethods,sep="_"),of)) # AUC spreadsheet