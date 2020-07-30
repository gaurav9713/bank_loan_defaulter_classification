rm(list = ls(all = T))
setwd("C:/Users/Gaurav/Desktop/Project 1")

#Reading the preprocessed data 

loan_data = read.csv("loan_data_calc.csv", header = T)


#Divide data into train and test using stratified sampling method
install.packages("caret")
require(caret)
set.seed(1234)
train.index = createDataPartition(loan_data$default, p = .80, list = FALSE)
train = loan_data[ train.index,]
test  = loan_data[-train.index,]


#Logiatic regression 

logit_model = glm(default ~ ., data = train, family = "binomial")

#summary of the model
summary(logit_model)

#predict using logistic regression
logit_Predictions = predict(logit_model, newdata = test, type = "response")

#convert prob
logit_Predictions = ifelse(logit_Predictions > 0.5, 1, 0)


##Evaluate the performance of classification model
ConfMatrix_logit = table(test$default, logit_Predictions)

confusionMatrix(ConfMatrix_logit)

TN = ConfMatrix_logit[2,2]
FN = ConfMatrix_logit[2,1]
TP = ConfMatrix_logit[1,1]
FP = ConfMatrix_logit[1,2]

#False Negative rate
FNR_logit = (FN*100)/(FN+TP)
Accuracy_logit = ((TP+TN)*100)/(TN+TP+FN+FP) 






## Decision Tree
require(rpart)
install.packages("C50")
require(C50)
install.packages("e1071")
require(e1071)

loan_data$default <- as.factor(loan_data$default)
str(loan_data$default)
train.index = createDataPartition(loan_data$default, p = .80, list = FALSE)
train = loan_data[ train.index,]
test  = loan_data[-train.index,]
C50_model = C5.0(default ~., train, trials = 100, rules = TRUE)

#Summary of DT model
summary(C50_model)
length(loan_data)
#write rules into disk
write(capture.output(summary(C50_model)), "c50Rules.txt")

#Lets predict for test cases
C50_Predictions = predict(C50_model, test[,-10], type = "class")

##Evaluate the performance of classification model
ConfMatrix_C50 = table(test$default, C50_Predictions)
confusionMatrix(ConfMatrix_C50)

ConfMatrix_C50

TN = ConfMatrix_C50[2,2]
FN = ConfMatrix_C50[2,1]
TP = ConfMatrix_C50[1,1]
FP = ConfMatrix_C50[1,2]

#False Negative rate
FNR_tree = (FN*100)/(FN+TP) 
Accuracy_tree = ((TP+TN)*100)/(TN+TP+FN+FP) 




## Random Forest
install.packages("randomForest")
require(randomForest)
RF_model = randomForest(default ~ ., train, importance = TRUE, ntree = 150)

#Presdict test data using random forest model
RF_Predictions = predict(RF_model, test[,-10])

##Evaluate the performance of classification model
ConfMatrix_RF = table(test$default, RF_Predictions)
confusionMatrix(ConfMatrix_RF)

confusionMatrix(ConfMatrix_RF)

TN = ConfMatrix_RF[2,2]
FN = ConfMatrix_RF[2,1]
TP = ConfMatrix_RF[1,1]
FP = ConfMatrix_RF[1,2]

#False Negative rate
FNR_RF = (FN*100)/(FN+TP) 
Accuracy_RF = ((TP+TN)*100)/(TN+TP+FN+FP) 







##Naive Bayes
NB_model = naiveBayes(default ~ ., data = train)

#predict on test cases #raw
NB_Predictions = predict(NB_model, test[,1:9], type = 'class')

#Look at confusion matrix
Confmatrix_NB = table(observed = test[,10], predicted = NB_Predictions)
confusionMatrix(Confmatrix_NB)

Confmatrix_NB

TN = Confmatrix_NB[2,2]
FN = Confmatrix_NB[2,1]
TP = Confmatrix_NB[1,1]
FP = Confmatrix_NB[1,2]

#False Negative rate
FNR_NB = (FN*100)/(FN+TP) 
Accuracy_NB = ((TP+TN)*100)/(TN+TP+FN+FP) 







##KNN Implementation
library(class)

#Predict test data
KNN_Predictions = knn(train[, 1:9], test[, 1:9], train$default, k = 5)

#Confusion matrix
Conf_matrix_KNN = table(KNN_Predictions, test$default)


confusionMatrix(Conf_matrix_KNN)

TN = Conf_matrix_KNN[2,2]
FN = Conf_matrix_KNN[2,1]
TP = Conf_matrix_KNN[1,1]
FP = Conf_matrix_KNN[1,2]

#False Negative rate
FNR_KNN = (FN*100)/(FN+TP) 
Accuracy_KNN = ((TP+TN)*100)/(TN+TP+FN+FP) 
