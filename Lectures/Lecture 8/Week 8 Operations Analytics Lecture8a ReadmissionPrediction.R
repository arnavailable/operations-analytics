# MGMTMSA 408, Operations Analytics
# Lecture 8: Readmission Prediction
# In-class example with artificial readmission data.

####### UNDERSTANDING THE DATA ######

# Read the data 
readm = read.csv("cases_small_v2.csv", stringsAsFactors = T)

# Let's take a look at the data:
str(readm)
summary(readm)

# How many patients are there in total?
nrow(readm)

# How many went to the ED within 30 days of their surgery?
table(readm$PostAdmissionEDVisit)

# For how many did their subsequent ED visit (possibly not in the next 30 days)
# result in being admitted?
table(readm$TransferFromED)

# How many went to the ED within 30 days of surgery, and
# then were transfered to a non-ED location?
table(readm$PostAdmissionEDVisit, readm$TransferFromED)

# Let us create our actual dependent variable:
readm$EDReadmission = readm$PostAdmissionEDVisit * readm$TransferFromED



####### HANDLING MISSING DATA ######

# In our data set, we have a number of variables that have missing values. 
z = function(a)
{
  return(sum(is.na(a)))
}
which(sapply(readm,z) > 0)

# There is no one way of handling missing data, but let us see a few different approaches.


# Approach #1: replace with mean/mode of non-missing data.
# One way to handle missingness is to replace the missing entries with the mean.
# Let's do this for BMI, Weight and Age:
mean(readm$BMI, na.rm = T) # Average BMI over patients with non-missing entries
mean(readm$Weight, na.rm = T) # Average Weight over patients with non-missing entries
mean(readm$Age, na.rm = T) # Average Age over patients with non-missing entries

# Replace the missing values with the means:
readm$BMI[ is.na(readm$BMI) ] = mean(readm$BMI, na.rm = T) 
readm$Weight[ is.na(readm$Weight) ] = mean(readm$Weight, na.rm = T) 
readm$Age[ is.na(readm$Age) ] = mean(readm$Age, na.rm = T) 

# For ASA score, let's replace missing value with most frequent one:
which.max(table(readm$ASAScore))
# Most frequent one is 3, so we do the following:
readm$ASAScore[ is.na(readm$ASAScore) ] = 3


# Approach #2: missingness means zero.
# In some cases, when something is missing, this can just be indicative of a zero value. 
# Let's take this approach with Smoking:
readm$Smoking[is.na(readm$Smoking)] = 0


# Approach #3: add an indicator for missingness.
# Sometimes, a variable being missing from an observation can be useful information.
# Let's take this approach with the Alcohol variable.

readm$Alcohol_NA = as.numeric(is.na(readm$Alcohol)) # Create a 0/1 variable to indicate that the variable is missing
readm$Alcohol[is.na(readm$Alcohol)] = 0 # Fill in a value of zero for the missing values of Alcohol.


# Approach #4: use missing data imputation approaches
# There are more sophisticated approaches for filling in missing data,
# where the essential idea is to use the columns that have values to predict
# the missing values. One such package is mice (Multivariate Imputation by Chained Equations).

# Let's install it first:
# install.packages("mice")
library(mice)

# Now let's run it. 
# This function is random, so let's set our seed so that our results can be reproduced...
set.seed(55)
readm_surgery = readm[,c("Surgery_IVColloids", "Surgery_BloodLoss", "Surgery_DurationMinutes", "Surgery_BloodTransfused", "Surgery_IVFluids")]
readm_imputation = mice(readm_surgery)
readm_surgery_completed = complete(readm_imputation)

readm$Surgery_IVColloids = readm_surgery_completed$Surgery_IVColloids
readm$Surgery_IVFluids = readm_surgery_completed$Surgery_IVFluids
readm$Surgery_BloodLoss = readm_surgery_completed$Surgery_BloodLoss
readm$Surgery_DurationMinutes = readm_surgery_completed$Surgery_DurationMinutes
readm$Surgery_BloodTransfused = readm_surgery_completed$Surgery_BloodTransfused


# Some care is needed with this method, because it uses all of the available variables in the imputed 
# data frame to do the imputation. It could, for example, use the dependent variable, or the PatientID.

# Before we continue, let's save the current data frame:
readm.original = readm

# (We will use it again later.)
# Let's drop the two precursor variables we used to calculate the dependent variable,
# as well as the patient ID:
readm$PatientID <- NULL
readm$PostAdmissionEDVisit <- NULL
readm$TransferFromED <- NULL



####### BUILDING AN INITIAL MODEL #######

# Load caTools
library(caTools)

# Let's all set our random number generator to the same seed,
# so we obtain the same results
set.seed(88)

# Now perform the split.
spl = sample.split(readm$EDReadmission, SplitRatio = 0.8)
# Usually, we want between 50% and 80% of the data in the training set.

# Take a look at spl:
spl

# Now split the data:
readm.train = subset(readm, spl == TRUE)
readm.test = subset(readm, spl == FALSE)

# Our first model will be a logistic regression model.
readm.glm = glm(EDReadmission ~ ., data = readm.train, family = "binomial")

# Examine the coefficients
summary(readm.glm)


####### OUT OF SAMPLE ACCURACY #######

# Let's now make predictions on our test set.
readm.predict = predict( readm.glm, newdata = readm.test, type = "response")

# We are going to take our predictions, and threshold them according to the 
# threshold of 0.2. 
# Confusion matrix:
confMat = table( readm.test$EDReadmission , readm.predict > 0.2 )
confMat

# Calculate the accuracy:
accuracy = sum( diag(confMat)) / nrow(readm.test)
accuracy

# We get a very high accuracy -- this is great! 
# Or is it? 

# Let's now get the baseline accuracy.
# First, look at occurrence of EDReadmission in training set:
table(readm.train$EDReadmission)
# Most frequent class is 0
# => Baseline model should predict zero (no ED readmission)

# Now look at test set:
table(readm.test$EDReadmission)
baseline_accuracy = 567 / (567 + 33)
baseline_accuracy


####### ROC CURVES / OUT OF SAMPLE AUC #########

# To produce an ROC curve / compute AUC, we'll
# need the ROCR package:
# install.packages("ROCR")
library(ROCR)

# The next three commands will give us the ROC curve
ROCpred = prediction(readm.predict, readm.test$EDReadmission)
ROCperf = performance(ROCpred, "tpr", "fpr")
# ROCperf = performance(ROCpred, "tnr", "fnr")
plot(ROCperf, main = "Receiver Operator Characteristic Curve")

# We can also plot the ROC curve with colors to indicate
# the thresholds.

plot(ROCperf, main = "Receiver Operator Characteristic Curve", 
     colorize=TRUE, print.cutoffs.at=seq(0,1,by=0.1), text.adj=c(-0.2,1.7))


# This last command will compute the AUC
AUC = as.numeric(performance(ROCpred, "auc")@y.values)
AUC

# We obtain an AUC of about 0.68 -- not great.
# Next time we will see how we can do better, by using better methods
# and using more data. 



