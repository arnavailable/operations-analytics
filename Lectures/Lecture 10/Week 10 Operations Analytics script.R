# MGMTMSA 408, Operations Analytics
# Lecture 10: Matrix Completion
# In-class example with MovieLens data.

####### LOADING THE DATA ######

# Load the dataset
ratings.df <- read.csv("/Users/arnavgarg/Documents/GitHub/operations-analytics/Lectures/Lecture 10/ratings.csv")
nrow(ratings.df)

# Examine the structure:
str(ratings.df)
summary(ratings.df)

# Load also movie-to-id-df.csv, which gives us the title of each movie.
titles.df <- read.csv("/Users/arnavgarg/Documents/GitHub/operations-analytics/Lectures/Lecture 10/movie_to_id.csv")

####### SPLITTING THE DATA ######

# Split the data 80-20
set.seed(20)
library(caTools)
spl <- sample.split(ratings.df$rating, SplitRatio = 0.8)
ratings.train <- ratings.df[spl, ]
ratings.test <- ratings.df[!spl, ]

# Create the sparse matrix.
# softImpute allows you to give it an incomplete matrix in several ways.
# One way is as a matrix with NAs in it.
# Another way is using softImpute's Incomplete class.
# This allows us to specify a matrix using just the indices of the entries.
# install.packages("softImpute")
library(softImpute)
ratings.train.mat <- Incomplete(i = ratings.train$customer.id, j = ratings.train$title.id, x = ratings.train$rating)
# ratings.train.mat

# Next, we are going to run the biScale function.
# biScale will normalize the rows and columns so that the rows and columns
# have zero mean, and that they are rescaled so that their standard deviation is 1.
# For our purposes, we will just use the centering capability of this function, and not
# use the scaling capability. (The scaling capability is helpful when the columns
# correspond to measurements that are on different scales.)
ratings.train.scaled <- biScale(ratings.train.mat, row.scale = F, col.scale = F, trace = T)
# ratings.train.scaled

# Now we will run softImpute.
# softImpute does randomly initialize the starting solution, so we need to set the seed.
# In the command below:
# - rank.max indicates the (maximum) dimension of the u and v vectors (the latent attribute space).
# - lambda indicates the value of the regularization parameter.
# - maxit indicates the maximum number of iterations
# Ideally, all of these values would be tuned using a validation set or k-fold cross-validation.
# For our purposes, we will choose arbitrary values for these.
set.seed(50)
# ratings.si = softImpute(ratings.train.scaled, rank.max = 100, lambda = 1, maxit = 1000, trace.it = T)
ratings.si <- softImpute(ratings.train.scaled, rank.max = 100, lambda = 7, maxit = 1000, trace.it = T)

# The ratings.si stores the completed matrix as three objects: u, d, v.
# u and v are the U and V matrices in the slides. d is the D matrix in the
# singular value decomposition; it is just a vector of values that appear along
# the diagonal of the D matrix.
# The reason for having d is that it allows u and v to be normalized to have unit length.
# For example:
sum(ratings.si$u[, 1]^2)
sum(ratings.si$v[, 3]^2)

# Something else to keep in mind is that rank.max specifies the largest allowable dimension
# of u and v, but the ultimate solution may not use all rank.max coordinates.
# We can check this by looking at ratings.si$d:
ratings.si$d

# We can also check this by looking at the u object in rating.si:
dim(ratings.si$u)



# Fill in the missing entries using the ratings.si object:
ratings.mat.imputed <- complete(ratings.train.mat, ratings.si)

# We now need to extract the ratings of the entries that we
# left out for the test set. Recall that ratings.test is in "long" form:
# each row corresponds to a different customer / movie pair.
# The loop below will help us to extract the corresponding completed entry
# from ratings.mat.imputed and line it up with the actual rating.
ratings.predict <- numeric(nrow(ratings.test))
for (ind in 1:nrow(ratings.test))
{
  ratings.predict[ind] <- ratings.mat.imputed[ratings.test$customer.id[ind], ratings.test$title.id[ind]]
}

# To evaluate the error, we can compute a few metrics.
# The first is R squared:
SSE <- sum((ratings.predict - ratings.test$rating)^2)
SST <- sum((ratings.test$rating - mean(ratings.train$rating))^2)
Rsq <- 1 - SSE / SST
Rsq

# We can also compute the root mean square error (RMSE), which
# is on average how far our predictions are from the actual ratings.
RMSE <- sqrt(SSE / nrow(ratings.test))
RMSE

# Note: RMSE here looks bad (on the order of 0.8), but for this application
# it is actually pretty decent. For the Netflix prize, Netflix's own algorithm
# had 0.9525, and the winning entry (which was an ensemble method) was 10.5% below
# this (~0.85).

# Lastly, we can also compute mean absolute error (MAE):
MAE <- mean(abs(ratings.predict - ratings.test$rating))
MAE


####### INTERPRETING THE RESULTS #######

# Besides the predictive model, it is also helpful to
# take a look at what the v vectors, which correspond to the
# latent features of the movie, look like.

# Extract v into its own object:
v <- ratings.si$v

# v has a row per movie, and a column for each latent dimension.
# Stick on the movie titles as the row names:
rownames(v) <- titles.df$title

# Now, let's examine each coordinate of the v vectors.
# We will do this by looking at the highest 10 values and lowest 10
# values of each coordinate, and which movies attain those values.

# Coordinate: 1
head(sort(v[, 1]), n = 10)
tail(sort(v[, 1]), n = 10)

# Coordinate 2:
head(sort(v[, 2]), n = 10)
tail(sort(v[, 2]), n = 10)

# Coordinate 3:
head(sort(v[, 3]), n = 10)
tail(sort(v[, 3]), n = 10)

# Coordinate 4:
head(sort(v[, 4]), n = 10)
tail(sort(v[, 4]), n = 10)

# Coordinate 5:
head(sort(v[, 5]), n = 10)
tail(sort(v[, 5]), n = 10)



# Another way of understanding our movies is to plot two of their coordinates
# on a scatter plot. (This is similar to biplots used in principal components analysis.)
plot(v[, 1], v[, 2])

# We can also plot the text of each movie's title on this plot (using
# the commented out command below), but with 2000 movies it's tricky to read.
# text( v[,1], v[,2], rownames(v))

# Instead, let's look at the extreme movies, which are the ones away
# from the center. We can do this by selecting those movies where the
# corresponding "point" is at least 0.09 units away from the center:
inds <- v[, 1]^2 + v[, 2]^2 >= 0.09^2
v_extremes <- v[inds, ]

# Now create the same plot:
plot(v_extremes[, 1], v_extremes[, 2])
text(v_extremes[, 1], v_extremes[, 2], rownames(v_extremes))



# Do again for coords 1 and 3 (at least 0.10 away from center).
# (NB: 0.10 was not chosen for any special reason, other than to produce
# a legible plot!)
inds <- v[, 1]^2 + v[, 3]^2 >= 0.10^2
v_extremes <- v[inds, ]
plot(v_extremes[, 1], v_extremes[, 2])
text(v_extremes[, 1], v_extremes[, 2], rownames(v_extremes))


# Do again for coords 2 and 3 (at least 0.10 away from center).
inds <- v[, 2]^2 + v[, 3]^2 >= 0.10^2
v_extremes <- v[inds, ]
plot(v_extremes[, 1], v_extremes[, 2])
text(v_extremes[, 1], v_extremes[, 2], rownames(v_extremes))
