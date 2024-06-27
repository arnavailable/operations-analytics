# MGMTMSA 408, Operations Analytics
# Lecture 8: Readmission Prediction
# In-class example with artificial readmission data.

# Note: should run Lecture8a_ReadmissionPrediction.R first, and then run
# the line below (needed as we will be doing some joins that use the PatientIDs):
readm = readm.original

####### INCORPORATING LABS #########

labs = read.csv("labs_small.csv", stringsAsFactors = FALSE)

# Let's see what is in the labs 
head(labs)

# Create a new column, which pastes the lab name and the TypeID. 
labs$LabTypeName = paste( gsub( " ", "_", labs$LabName), labs$LabTypeID, sep ='_')

# We are now going to use a function dcast. This function takes a data frame in 
# "long" format, and gives us a data frame in wide format. It is specified with a formula;
# the variables on the left correspond to the rows of the new data frame, and the variables on 
# the right correspond to the columns. For each combination of the two sets of variables 
# - in our case, PatientID and LabTypeName -- it will apply an aggregation function (max)
# to another variable in labs (this is the value.var, which is Delta). 
# install.packages("reshape2")
library(reshape2)
labs_wide = dcast( labs, PatientID ~ LabTypeName, max, value.var = "Delta" )

# Let's take a look at the first few rows:
head(labs_wide)

# We see some -Inf's -- why did this happen?

# When applying dcast with max, we may encounter combinations of PatientId and LabTypeName
# for which the array of values in labs is empty. max() with an empty array returns -Inf:
max( numeric(0) )

# To fix this, let's zero out those values.
labs_wide[labs_wide == -Inf] = 0

# Lastly, let's also add a suffix to these new variables:
names(labs_wide)[-1] = paste(names(labs_wide)[-1], "_maxdelta", sep ='')


# Now add these columns to our original data frame:
library(dplyr)
readm.wlabs = left_join(readm, labs_wide, by = c("PatientID" = "PatientID"))
# readm.wlabs = left_join(readm.wlabs, labs_wide_dummy, by = c("PatientID" = "PatientID"))

# Let's just check that things look OK:
summary(readm.wlabs)

# Oh no! We have new NAs. Why did this happen?

# Well - for some patients, they simply have no lab records.
# We can handle this in various ways, but for simplicity, let's zero out
# these variables. Since we are doing this for all of these new variables,
# we can just use the following command:
readm.wlabs[ is.na(readm.wlabs) ] = 0

# Check again:
summary(readm.wlabs)


# Terrific! 
# Let's drop PatientID, PostAdmissionEDVisit, TransferFromED again, and continue with our business.
readm.wlabs$PatientID <- NULL
readm.wlabs$PostAdmissionEDVisit <- NULL
readm.wlabs$TransferFromED <- NULL


####### BUILDING A STRONGER MODEL WITH LABS #########

# Split our data again:
library(caTools)
set.seed(88)
spl = sample.split(readm$EDReadmission, SplitRatio = 0.8)
readm.wlabs.train = subset(readm.wlabs, spl == TRUE)
readm.wlabs.test = subset(readm.wlabs, spl == FALSE)

# Run vanilla logistic regression:
readm.wlabs.glm = glm(EDReadmission ~ ., data = readm.wlabs.train, family = "binomial")

# Examine the coefficients
summary(readm.wlabs.glm)

# Now let's look at the test set.
readm.test.predict = predict( readm.wlabs.glm, newdata = readm.wlabs.test, type = "response" )
# Confusion matrix:
confMat = table( readm.wlabs.test$EDReadmission, readm.test.predict > 0.5 )
confMat

# Calculate the accuracy:
accuracy = sum( diag(confMat)) / nrow(readm.wlabs.test)
accuracy

# Let's calculate the AUC:
library(ROCR)

# The next three commands will give us the ROC curve
ROCpred = prediction(readm.test.predict, readm.wlabs.test$EDReadmission)
ROCperf = performance(ROCpred, "tpr", "fpr")
plot(ROCperf, main = "Receiver Operator Characteristic Curve")
AUC = as.numeric(performance(ROCpred, "auc")@y.values)
AUC



####### LASSO LOGISTIC REGRESSION MODEL #######

# The next model that we are going to build is a lasso / L1 regularized 
# logistic regression model.
# To build it, we are going to need the glmnet and glmnetUtils packages:
library(glmnet)
library(glmnetUtils)

# glmnet implements regularized regression, while glmnetUtils provides
# nice formula-based functions for using glmnet's functions

# The following command runs a lasso logistic regression with 5-fold cross validation.
# Remember to set the seed beforehand, because k-fold cross validation will always use
# the random number generator.
set.seed(50)
readm.wlabs.glmnet = cv.glmnet( EDReadmission ~ ., data = readm.wlabs.train, family = "binomial", nfolds = 5, standardize = TRUE)

# Recall from your earlier class on machine learning that k-fold cross validation is a way to help us to
# choose parameters. In the case of lasso, the parameter that we need to decide on is lambda. 

# We can see how the cross-validated binomial deviance (= 2 * negative log likelihood) varies as a function of 
# lambda:
plot(readm.wlabs.glmnet)

# On the plot:
# - The bottom axis indicates the value of lambda on a log scale.
# - The values on the top indicate the sparsity / support size of the logistic regression model,
# i.e., how many coefficients are non-zero at each value of lambda.
# - The y-axis gives the binomial deviance.
# - The error bars correspond to one standard error.
#
# There are two vertical lines. One indicates at the minimum of the cross-validated deviance curve,
# and is denoted by "lambda.min" within glmnet functions/objects. You can access it as:
readm.wlabs.glmnet$lambda.min
log(readm.wlabs.glmnet$lambda.min)

# The other vertical line indicates the highest value of lambda, such that the deviance is within
# 1 standard error of the minimum deviance (the one attained at lambda.min). The lambda of this
# second vertical line is indicated by lambda.1se. 

readm.wlabs.glmnet$lambda.1se
log(readm.wlabs.glmnet$lambda.1se)

# The rationale for lambda.1se is that in tuning parameters with cross-validation, it is very
# common to encounter situations where the cross-validated error curve is very flat around the 
# parameter value that minimizes the curve. In this type of situation, while there is one "best" 
# value of the parameter, there are many values of the parameter which are very close to being optimal.
# In such a situation, it can be more desirable to choose the parameter value that is "close enough"
# to the true minimizer, while yielding the smallest model. In our case, setting lambda higher gives
# us fewer non-zero coefficients, so lambda.1se will be a larger lambda value than lambda.min.

# Let us first see what the predictions are like with lambda.min, and then examine the 
# coefficients. We can make predictions on new data using predict(), and specifying the 
# s parameter to be "lambda.min"

readm.test.predict = predict( readm.wlabs.glmnet, newdata = readm.wlabs.test, type = "response", s = "lambda.min" )

# Accuracy:
confMat = table( readm.wlabs.test$EDReadmission, readm.test.predict > 0.5 )
confMat
accuracy = sum( diag(confMat)) / nrow(readm.wlabs.test)
accuracy

# AUC:
ROCpred = prediction(readm.test.predict, readm.wlabs.test$EDReadmission)
ROCperf = performance(ROCpred, "tpr", "fpr")
plot(ROCperf, main = "Receiver Operator Characteristic Curve")
AUC = as.numeric(performance(ROCpred, "auc")@y.values)
AUC

# We have an improvement! 

# Let's see the coefficients:

lasso.coeffs.min = coefficients(readm.wlabs.glmnet, s = "lambda.min")
lasso.coeffs.min

# This is hard to examine, but it is easy to focus on just the non-zero ones:
rownames(lasso.coeffs.min)[ which(lasso.coeffs.min != 0)]

# Let's now see what happens when we use lambda.1se
readm.test.predict = predict( readm.wlabs.glmnet, newdata = readm.wlabs.test, type = "response", s = "lambda.1se" )

# Accuracy:
confMat = table( readm.wlabs.test$EDReadmission, readm.test.predict > 0.5 )
confMat
accuracy = sum( diag(confMat)) / nrow(readm.wlabs.test)
accuracy

# AUC:
ROCpred = prediction(readm.test.predict, readm.wlabs.test$EDReadmission)
ROCperf = performance(ROCpred, "tpr", "fpr")
plot(ROCperf, main = "Receiver Operator Characteristic Curve")
AUC = as.numeric(performance(ROCpred, "auc")@y.values)
AUC
# Our AUC is not as high as before.

# Let's see the coefficients:
lasso.coeffs.1se = coefficients(readm.wlabs.glmnet, s = "lambda.1se")
lasso.coeffs.1se
rownames(lasso.coeffs.1se)[ which(lasso.coeffs.1se != 0)]



####### CLASSIFICATION TREE MODEL #######
# We will now see how to build a classification tree model for this data. 
# To do this, let's load the rpart package and then run the command below it:
library(rpart)
readm.wlabs.rpart = rpart( EDReadmission ~ ., data = readm.wlabs.train, method = "class")

# The first two arguments are the same as for lm() and glm().
# The third argument tells CART to build a classification tree.
# If the dependent variable is numeric and we leave method="class" out, then
# rpart will by default build a regression tree. In this case, EDReadmission
# is a numeric variable, so either we need to give method = "class", or
# we need to convert EDReadmission to a factor before hand. 

# To visualize the tree, we can use the rpart.plot package, which provides
# the function prp for plotting trees.
library(rpart.plot)
prp(readm.wlabs.rpart)

# prp provides an "extra" parameter, which allows us to modify what is displayed.
# Here are a few examples:

# extra = 6 -> plot the proportion of observations in the leaf that belong to class 1.
prp(readm.wlabs.rpart, extra= 6)

# Another example: extra = 1 will plot the raw number of observations
# belonging to each class that fall in each leaf.
prp(readm.wlabs.rpart, extra= 1)

# (For more information, run ?prp)

# Let's now see the effect of cp.
# As mentioned in the slides, the cp parameter controls how deep the tree is.
# The default is cp = 0.01. Let's see what happens with cp = 0.001:
readm.wlabs.rpart = rpart( EDReadmission ~ ., data = readm.wlabs.train, method = "class", cp = 0.001)
prp(readm.wlabs.rpart)

# What do you think should happen when cp = 0?
readm.wlabs.rpart = rpart( EDReadmission ~ ., data = readm.wlabs.train, method = "class", cp = 0, minbucket = 0, minsplit = 0)
prp(readm.wlabs.rpart)

# Now let's go in the other direction. Suppose we set cp = 0.015:
readm.wlabs.rpart = rpart( EDReadmission ~ ., data = readm.wlabs.train, method = "class", cp = 0.015)
prp(readm.wlabs.rpart)

# Now try cp = 0.02:
readm.wlabs.rpart = rpart( EDReadmission ~ ., data = readm.wlabs.train, method = "class", cp = 0.02)
prp(readm.wlabs.rpart)

# What model does this tree correspond to? 


# Going forward, let's use the cp = 0.015 model. 
readm.wlabs.rpart = rpart( EDReadmission ~ ., data = readm.wlabs.train, method = "class", cp = 0.015)
prp(readm.wlabs.rpart)

readm.test.predict = predict( readm.wlabs.rpart, newdata = readm.wlabs.test )
readm.test.predict = readm.test.predict[,2]

# Accuracy:
confMat = table( readm.wlabs.test$EDReadmission, readm.test.predict > 0.5 )
confMat
accuracy = sum( diag(confMat)) / nrow(readm.wlabs.test)
accuracy

# AUC:
library(ROCR)
ROCpred = prediction(readm.test.predict, readm.wlabs.test$EDReadmission)
ROCperf = performance(ROCpred, "tpr", "fpr")
plot(ROCperf, main = "Receiver Operator Characteristic Curve")
AUC = as.numeric(performance(ROCpred, "auc")@y.values)
AUC


####### RANDOM FOREST MODEL #######

# The last type of model that we will consider is the random forest model.
# Let us all first install the randomForest package, and then load it.
# Note: please do this, even if you may already have it installed. 
install.packages("randomForest")
library(randomForest)

# Next, we will estimate the random forest. Since randomForest will draw random numbers
# (in order to do bagging and to apply the random subspace method), we need to set the seed.
set.seed(101)

# Now, run the following command:
readm.wlabs.rf = randomForest( as.factor(EDReadmission) ~ ., data = readm.wlabs.train)

# A note about the above command: within the formula, we are converting EDReadmission to a factor.
# This is needed because without it, randomForest defaults to doing regression when the dependent variable
# is numeric. For example, you will see a warning when you try this command:
readm.wlabs.rf = randomForest( EDReadmission ~ ., data = readm.wlabs.train)

# Next, let's make some predictions.
# The following command will get the actual classifications from the random forest:
readm.test.predict = predict( readm.wlabs.rf, newdata = readm.wlabs.test)
head(readm.test.predict)

# Since we need a probability in order to calculate AUC, we will use predict in a slightly different way:
readm.test.predict = predict( readm.wlabs.rf, newdata = readm.wlabs.test, type = "prob")
readm.test.predict = readm.test.predict[,2]
head(readm.test.predict)


# Accuracy:
confMat = table( readm.wlabs.test$EDReadmission, readm.test.predict > 0.5 )
confMat
accuracy = sum( diag(confMat)) / nrow(readm.wlabs.test)
accuracy

# AUC:
library(ROCR)
ROCpred = prediction(readm.test.predict, readm.wlabs.test$EDReadmission)
ROCperf = performance(ROCpred, "tpr", "fpr")
plot(ROCperf, main = "Receiver Operator Characteristic Curve")
AUC = as.numeric(performance(ROCpred, "auc")@y.values)
AUC


# Lastly, let us see how we can examine the model. 
# Let's calculate the impurity importance plot:
varImpPlot(readm.wlabs.rf)

# Note that this only provides the impurity importances.
# To get the permutation importances, we actually need to add an
# additional input parameter, importance = TRUE, to the randomForest function:
set.seed(101)
readm.wlabs.rf = randomForest( as.factor(EDReadmission) ~ ., data = readm.wlabs.train, importance = TRUE)

# One important note about this command. The permutation importance metric requires the
# random number generator; as a result, the random forest you get with importance = TRUE
# will not be the same as the one when you omit it (and it defaults to FALSE). 

# Generate both plots now:
varImpPlot(readm.wlabs.rf)w

