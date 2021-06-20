#######################################################################################
################################# Prepare Environment #################################
#######################################################################################
# Install and load Packages
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
library(tidyverse)
library(caret)
library(data.table)
library(dplyr)
library(lubridate)


# Download required data sources
# download data file
dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

# Prepare raw data from data sources
# unzip and read ratings from file into columns
ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

# reads movies data file and extracts columns
movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# only for code reruns
save(ratings,file = 'rda/ratings.rda')
save(movies,file = 'rda/movies.rda')

# only for code reruns
load(file='rda/ratings.rda')
load(file='rda/movies.rda')

# Explore ratings and movies
str(ratings)
str(movies)

# Wrangle dataset
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

str(movies)

movielens <- left_join(ratings, movies, by = "movieId")
str(movielens)

# Glance Movie Lens Data
movielens %>% head()


# Prepare data partition for validation and training
# if using R 3.5 or earlier, use `set.seed(1)`
set.seed(1, sample.kind="Rounding")  

# Validation set will be 10% of MovieLens data
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

str(edx)
str(temp)

# Make sure userId and movieId in validation set are also in edx set
# Test set (temp) would be useful to have only those rows movies, users, which were in training set. So we keep only those observations and store in validation.
validation <- temp %>% semi_join(edx, by = "movieId") %>% semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
# The removed rows from validation set can be used for training. So we add them back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

str(removed)

# removing irrelevant objects
rm(dl, ratings,test_index, temp, movielens, removed)

# The partitioned data were named edx and validation, 
# where edx was our training data set and validation was our validation data set, 
# which we kept aside as planned earlier. 

data.frame(data_set=c("Training data set","Validation data set"),name=c('edx','validation'), count=c(nrow(edx),nrow(validation))) %>% knitr::kable()

#######################################################################################
################################### Basic analysis ####################################
#######################################################################################

# Let's look at how users have rated movies
# If we look at ratings, users rated few movies
info <- edx %>% summarise(users_count=n_distinct(userId), movies_count=n_distinct(movieId)) 
total_ratings <- edx %>% nrow()

average_movies_rated_per_user <- total_ratings/info$users_count

# So average number of movies rated per user
combine_words(c('Average number of movies rated per user',average_movies_rated_per_user),and=': ')

# Let's see number of unique movies and users in our training data set.
info %>% knitr::kable()

# Cleanup irrelvant variables
rm(average_movies_rated_per_user,info, total_ratings)

# To visualise how spare the dataset is
# we randomly selected 50 users and 50 movies with atleast one rating made by the selected user group and plotted a graph.
random50userIds <- edx %>% distinct(userId) %>% slice_sample(n=50) %>% pull(userId)
edx %>%  filter(userId %in% random50userIds) %>% select(userId,movieId,rating) %>%  mutate(rating=1) %>% spread(movieId,rating) %>% select(sample(ncol(.),50)) %>% as.matrix() %>% t(.) %>% image(1:50, 1:50,. , xlab="Movies", ylab="Users")
abline(h=0:50+0.5, v=0:50+0.5, col = "grey")

rm(random50userIds)

#######################################################################################
################### Parition edx dataset into test and training set ###################
#######################################################################################

# Create subset of data for fine tuning and model selection
# will use multiple partitions to remove any randomness from our tuning parameters 
# and to get an appopriate estimate of performance for our incremental model
y <- edx$rating
set.seed(1, sample.kind = 'Rounding')
test_indices_list <- createDataPartition(y,times=5,p=0.2,list=TRUE)

rm(y)

# Let's define performance metric RMSE in our code
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

#######################################################################################
############################ Model training and evaluation ############################
#######################################################################################

# Brute force model
# retrieve results for all partitions of data sets, while not required here due to stratified
# cross validations but retained if we want to update the code and it can be quickly done.
mean_of_partion_list <- sapply(1:5, function(index) {
  test_indices <- test_indices_list[[index]]
  train_set <- edx[-test_indices,]
  
  y_train_set <- train_set$rating
  
  total_mean <- mean(y_train_set)
}
)

total_mean <- mean(mean_of_partion_list)
combine_words(c('Total mean',total_mean),and=': ')
rm(mean_of_partion_list)

# Performance Evaluation
rmses <- sapply(1:5, function(index) {
  test_indices <- test_indices_list[[index]]
  train_set <- edx[-test_indices,]
  test_set <- edx[test_indices,]
  
  x_train_set <- train_set %>% select(userId,movieId,genres)
  
  # Before we use test_set, it needs to be filtered as it could have rows which have movies, users not present in training data.
  # So, we use only that test data which has movieId, userId in the training data
  test_set <- test_set %>% semi_join(x_train_set,by="movieId") %>% semi_join(x_train_set,by="userId")
  
  x_test_set <- test_set %>% select(userId,movieId,genres)
  y_test_set <- test_set$rating
  RMSE(y_test_set,total_mean)
}
)
performance_metric <- data.frame(model_name='Brute force model',rmse=mean(rmses))

performance_metric %>% knitr::kable()
rm(rmses)

# Movie effects
# movies may be rated differently
# Explore average rating per movie
edx %>% group_by(movieId) %>% summarise(avg=mean(rating)) %>% ggplot(aes(avg)) + geom_histogram(binwidth = 1) + xlab('Average Movie Rating') + geom_vline(xintercept=total_mean,linetype=6, color = "red",size=0.6)

# Check if some movies have very less number of ratings requiring regularisation
edx %>% dplyr::count(movieId) %>% ggplot(aes(log10(n))) + geom_histogram(binwidth=0.15) + xlab('Number of ratings per movie [log10(n) scale]')

# Looks like in this case, regularisation is required. So let's prepare data set for the same.
# Split training data for lambda computation
lambdas_per_effect <- data.frame(effect_name=character(0),lambda=numeric(0))

prepare_lambda_data_set <- function(data_partition_index) {
  test_indices <- test_indices_list[[data_partition_index]]
  train_set <- edx[-test_indices,]
  
  x_train_set <- train_set %>% select(userId,movieId,genres)
  y_train_set <- train_set$rating
  
  
  set.seed(1,sample.kind='Rounding')
  test_lambda_indices <- createDataPartition(train_set$rating,p=0.2,times=1,list=FALSE)
  train_lambda_set <- train_set[-test_lambda_indices,]
  test_lambda_set <- train_set[test_lambda_indices,]
  
  # Ensure that test_lambda_set only has movies and users present in train_lambda_set
  test_lambda_set <- test_lambda_set %>% semi_join(train_lambda_set,by="movieId") %>% semi_join(x_train_set,by="userId")
  list(train=train_lambda_set,test=test_lambda_set)
}

# Compute lambda for regularization. This is executed across all the partitions
lambdas <- seq(0,10,0.25)

lambda_to_use_per_partition <- sapply(1:5, function(index) {
  lambda_data_set <- prepare_lambda_data_set(index)
  train_lambda_set <- lambda_data_set[[1]]
  test_lambda_set <- lambda_data_set[[2]]
  
  lambda_rmses <- sapply(lambdas, function(lambda) {
    movie_effects <- train_lambda_set %>% group_by(movieId) %>% summarise(n=n(),b_i_hat=sum(rating - total_mean)/(n+lambda));
    predicted_ratings <- total_mean + test_lambda_set %>% left_join(movie_effects,by='movieId') %>% pull(b_i_hat)
    
    RMSE(test_lambda_set$rating,predicted_ratings) 
  })
  
  # selecting lambda leading to lowest RMSE
  lambda_to_use <- lambdas[which.min(lambda_rmses)]
  paste('Lambda to use: ', lambda_to_use, ', corresponding RMSE: ', lambda_rmses[which.min(lambda_rmses)])
  
  lambda_to_use
})

lambda_to_use_per_partition

# selecting lambda
lambda_to_use <- mean(lambda_to_use_per_partition)

lambdas_per_effect <- lambdas_per_effect %>% add_row(effect_name='movies',lambda=lambda_to_use)
rm(lambda_to_use_per_partition, lambdas)

# Post regularization of movie effect
# computed movie effects to be used later based on all paritions
movie_effects_list <- lapply(1:5, function(index) {
  test_indices <- test_indices_list[[index]]
  train_set <- edx[-test_indices,]
  
  y_train_set <- train_set$rating
  
  movie_effects <- train_set %>% group_by(movieId) %>% summarise(n=n(),b_i_hat=sum(rating - total_mean)/(n+lambda_to_use));
  movie_effects
}
)

me <- movie_effects_list[[1]]
for (index in 2:5) {
  me <- me %>% full_join(movie_effects_list[[index]],by='movieId', suffix=c("1",index))
}

me <- me %>% select(-n1,-n2,-n11,-n4,-n)
# as different partitions may not have exactly same data set combined the results
movie_effects <- tibble(me[,1], b_i_hat=rowMeans(me[,-1],na.rm = TRUE))

# to ensure we don't get random rmse due to data set. Rmse calculated based on all partitions
rmses <- sapply(1:5, function(index) {
  test_indices <- test_indices_list[[index]]
  train_set <- edx[-test_indices,]
  test_set <- edx[test_indices,]
  
  x_train_set <- train_set %>% select(userId,movieId,genres)
  
  # Before we use test_set, it needs to be filtered as it could have rows which have movies, users not present in training data.
  # So, we use only that test data which has movieId, userId in the training data
  test_set <- test_set %>% semi_join(x_train_set,by="movieId") %>% semi_join(x_train_set,by="userId")
  
  x_test_set <- test_set %>% select(userId,movieId,genres)
  y_test_set <- test_set$rating
  
  movie_effects <- train_set %>% group_by(movieId) %>% summarise(n=n(),b_i_hat=sum(rating - total_mean)/(n+lambda_to_use));
  
  predicted_ratings <- total_mean + test_set %>% left_join(movie_effects,by='movieId') %>% pull(b_i_hat)
  RMSE(y_test_set,predicted_ratings)
}
)

# Let's see how the movie effect varies.
movie_effects %>% ggplot(aes(b_i_hat)) + geom_histogram(bins=10,color=I('black'))

# Let's evaluate performance for this model
# y_u_i_hat = true_rating + b_i_hat
performance_metric <- performance_metric %>% add_row(model_name='With only movie effect model',rmse=mean(rmses))

performance_metric %>% knitr::kable()
rm(index, me, movie_effects_list, rmses, lambda_to_use)
# As we can see the evaluation metric i.e. RMSE improved considerably and therefore the movie effect was retained in our model.

# User effects
# look into how users rate movies differently 
# Users' rating pattern
edx %>% group_by(userId) %>% filter(n()>=100) %>% summarise(avg=mean(rating)) %>% ggplot(aes(avg)) + geom_histogram(bins=30,color=I('black')) + xlab('Average Rating Given By User')

# Check if some users have made less ratings requiring regularisation
edx %>% dplyr::count(userId) %>% ggplot(aes(log10(n))) + geom_histogram(binwidth=0.15) + xlab('Number of ratings per user [log10(n) scale]')

# Compute lambda for regularization for user effect
lambdas <- seq(0,10,0.25)

lambda_to_use_per_partition <- sapply(1:5, function(index) {
  lambda_data_set <- prepare_lambda_data_set(index)
  train_lambda_set <- lambda_data_set[[1]]
  test_lambda_set <- lambda_data_set[[2]]
  
  lambda_rmses <- sapply(lambdas, function(lambda) {
    user_effects <- train_lambda_set %>% left_join(movie_effects,by='movieId') %>% group_by(userId) %>% summarise(n=n(),b_u_hat=sum(rating - total_mean - b_i_hat)/(n+lambda))
    
    predicted_ratings <- test_lambda_set %>% left_join(movie_effects,by='movieId') %>% left_join(user_effects,by='userId') %>% mutate(pred=total_mean+b_i_hat+b_u_hat) %>% pull(pred)
    
    RMSE(test_lambda_set$rating,predicted_ratings) 
  })
  
  # selecting lambda leading to lowest RMSE
  lambda_to_use <- lambdas[which.min(lambda_rmses)]
  paste('Lambda to use: ', lambda_to_use, ', corresponding RMSE: ', lambda_rmses[which.min(lambda_rmses)])
  
  lambda_to_use
})

lambda_to_use_per_partition

# selecting lambda 
lambda_to_use <- mean(lambda_to_use_per_partition)

lambdas_per_effect <- lambdas_per_effect %>% add_row(effect_name='users',lambda=lambda_to_use)
rm(lambdas,lambda_to_use_per_partition)

# Post regularization of user effect
# So let's estimate the users'effect
# y_u_i_hat = true_rating + b_i_hat + b_u_hat

# computed user effects to be used later based on all paritions
user_effects_list <- lapply(1:5, function(index) {
  test_indices <- test_indices_list[[index]]
  train_set <- edx[-test_indices,]
  
  y_train_set <- train_set$rating
  
  user_effects <- train_set %>% left_join(movie_effects,by='movieId') %>% group_by(userId) %>% summarise(n=n(),b_u_hat=sum(rating - total_mean - b_i_hat)/(n+lambda_to_use))
  
  user_effects
}
)

# as different partitions may not have exactly same data set combined the results
ue <- user_effects_list[[1]]
for (index in 2:5) {
  ue <- ue %>% full_join(user_effects_list[[index]],by='userId', suffix=c("1",index))
}

ue <- ue %>% select(-n1,-n2,-n11,-n4,-n)
user_effects <- tibble(ue[,1], b_u_hat=rowMeans(ue[,-1],na.rm = TRUE))

# to ensure we don't get random rmse due to data set. Rmse calculated based on all partitions
rmses <- sapply(1:5, function(index) {
  test_indices <- test_indices_list[[index]]
  train_set <- edx[-test_indices,]
  test_set <- edx[test_indices,]
  
  x_train_set <- train_set %>% select(userId,movieId,genres)
  
  # Before we use test_set, it needs to be filtered as it could have rows which have movies, users not present in training data.
  # So, we use only that test data which has movieId, userId in the training data
  test_set <- test_set %>% semi_join(x_train_set,by="movieId") %>% semi_join(x_train_set,by="userId")
  
  x_test_set <- test_set %>% select(userId,movieId,genres)
  y_test_set <- test_set$rating
  
  user_effects <- train_set %>% left_join(movie_effects,by='movieId') %>% group_by(userId) %>% summarise(n=n(),b_u_hat=sum(rating - total_mean - b_i_hat)/(n+lambda_to_use))
  
  predicted_ratings <- test_set %>% left_join(movie_effects,by='movieId') %>% left_join(user_effects,by='userId') %>% mutate(pred=total_mean+b_i_hat+b_u_hat) %>% pull(pred)
  
  RMSE(y_test_set,predicted_ratings)
}
)

# Let's check how user effect varies across users
user_effects %>% ggplot(aes(b_u_hat)) + geom_histogram(bins=10)

# Let's evaluate performance for this model
# y_u_i_hat = true_rating + b_i_hat + b_u_hat
performance_metric <- performance_metric %>% add_row(model_name='With movie and user effect model',rmse=mean(rmses))

performance_metric %>% knitr::kable()
rm(ue, user_effects_list, index, lambda_to_use, rmses)
# As we can see the evaluation metric i.e. RMSE improved considerably and therefore the user effect was retained in our model.


# Genres effect
# Let's plot to see rating variation across genres
# while we could have used granular genres, using genres as it is gave use more precise control over prediction
edx %>% group_by(genres) %>%
  summarize(n = n(), avg = mean(rating), se = sd(rating)/sqrt(n())) %>%
  mutate(genres = reorder(genres, avg)) %>%
  ggplot(aes(x = genres, y = avg, ymin = avg - 2*se, ymax = avg + 2*se)) + 
  geom_point() +
  geom_errorbar() + 
  theme(axis.text.x =  element_blank()) + 
  xlab("Genres") + 
  ylab('Average rating of all movies for a given genre')

# Check if some genres have very less ratings requiring regularisation
edx %>% dplyr::count(genres) %>% ggplot(aes(log10(n))) + geom_histogram(binwidth=0.15) + xlab('Number of ratings across movies per genre [log10(n) scale]')

# Compute lambda for regularization of genre effect
lambdas <- seq(0,10,1)

lambda_to_use_per_partition <- sapply(1:5, function(index) {
  lambda_data_set <- prepare_lambda_data_set(index)
  train_lambda_set <- lambda_data_set[[1]]
  test_lambda_set <- lambda_data_set[[2]]
  
  lambda_rmses <- sapply(lambdas, function(lambda) {
    genre_effects <- train_lambda_set %>% left_join(movie_effects,by='movieId') %>% left_join(user_effects,by='userId') %>% group_by(genres) %>% summarise(n=n(),b_g_hat=sum(rating - total_mean-b_i_hat-b_u_hat)/(n+lambda))
    
    predicted_ratings <- test_lambda_set %>% left_join(movie_effects,by='movieId') %>% left_join(user_effects,by='userId') %>% left_join(genre_effects,by='genres') %>% mutate(pred=total_mean+b_i_hat+b_u_hat+b_g_hat) %>% pull(pred)
    
    RMSE(test_lambda_set$rating,predicted_ratings) 
  })
  
  # selecting lambda leading to lowest RMSE
  lambda_to_use <- lambdas[which.min(lambda_rmses)]
  paste('Lambda to use: ', lambda_to_use, ', corresponding RMSE: ', lambda_rmses[which.min(lambda_rmses)])
  
  lambda_to_use
})

lambda_to_use_per_partition

# selecting lambda 
lambda_to_use <- mean(lambda_to_use_per_partition)

lambdas_per_effect <- lambdas_per_effect %>% add_row(effect_name='genres',lambda=lambda_to_use)
rm(lambdas,lambda_to_use_per_partition)

# Post regularization of genre effect
# So let's estimate the genre'effect
# y_u_i_hat = true_rating + b_i_hat + b_u_hat + b_g_hat

# computed genres effects to be used later based on all paritions
genres_effects_list <- lapply(1:5, function(index) {
  test_indices <- test_indices_list[[index]]
  train_set <- edx[-test_indices,]
  
  y_train_set <- train_set$rating
  
  genre_effects <- train_set %>% left_join(movie_effects,by='movieId') %>% left_join(user_effects,by='userId') %>% group_by(genres) %>% summarise(n=n(),b_g_hat=sum(rating - total_mean-b_i_hat-b_u_hat)/(n+lambda_to_use))
  
  genre_effects
}
)

# as different partitions may not have exactly same data set combined the results
ge <- genres_effects_list[[1]]
for (index in 2:5) {
  ge <- ge %>% full_join(genres_effects_list[[index]],by='genres', suffix=c("1",index))
}

ge <- ge %>% select(-n1,-n2,-n11,-n4,-n)
genre_effects <- tibble(ge[,1], b_g_hat=rowMeans(ge[,-1],na.rm = TRUE))

# to ensure we don't get random rmse due to data set. Rmse calculated based on all partitions
rmses <- sapply(1:5, function(index) {
  test_indices <- test_indices_list[[index]]
  train_set <- edx[-test_indices,]
  test_set <- edx[test_indices,]
  
  x_train_set <- train_set %>% select(userId,movieId,genres)
  
  # Before we use test_set, it needs to be filtered as it could have rows which have movies, users not present in training data.
  # So, we use only that test data which has movieId, userId in the training data
  test_set <- test_set %>% semi_join(x_train_set,by="movieId") %>% semi_join(x_train_set,by="userId")
  
  x_test_set <- test_set %>% select(userId,movieId,genres)
  y_test_set <- test_set$rating
  
  genre_effects <- train_set %>% left_join(movie_effects,by='movieId') %>% left_join(user_effects,by='userId') %>% group_by(genres) %>% summarise(n=n(),b_g_hat=sum(rating - total_mean-b_i_hat-b_u_hat)/(n+lambda_to_use))
  
  predicted_ratings <- test_set %>% left_join(movie_effects,by='movieId') %>% left_join(user_effects,by='userId') %>% left_join(genre_effects,by='genres') %>% mutate(pred=total_mean+b_i_hat+b_u_hat+b_g_hat) %>% pull(pred)
  
  RMSE(y_test_set,predicted_ratings)
}
)

# Let's see how the genre effect varies.
genre_effects %>% ggplot(aes(b_g_hat)) + geom_histogram(bins = 100,color=I('black')) + scale_y_sqrt() + ylab('count [sqrt scale]')

# Let's evaluate performance for this model
# y_u_i_hat = true_rating + b_i_hat + b_u_hat + b_g_hat
performance_metric <- performance_metric %>% add_row(model_name='With movie, user and genre effect model',rmse=mean(rmses))

performance_metric %>% knitr::kable()
rm(ge,genres_effects_list,index,lambda_to_use,rmses)
# As we can see the evaluation metric i.e. RMSE improved and therefore the genre effect was retained in our model.



# Year of release effect
# extracting year of release
load(file='rda/movies.rda')
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

release_year <- sapply(movies$title,function(title){ str_match_all(title,'\\((\\d\\d\\d\\d)\\)')[[1]][1,2]})

release_year_per_movie <- movies %>% mutate(year_of_release=as.numeric(release_year)) %>% select(movieId,year_of_release)

# Example of extracted attribute
release_year_per_movie %>% head() %>% knitr::kable()

# Visualizing Avg ratings of movies released in an year vs year of release
edx %>% left_join(release_year_per_movie,by='movieId') %>% group_by(year_of_release) %>%
  summarize(n = n(), avg = mean(rating), se = sd(rating)/sqrt(n())) %>%
  ggplot(aes(x = year_of_release, y = avg, ymin = avg - 2*se, ymax = avg + 2*se)) + 
  geom_point() +
  geom_errorbar() + 
  theme(axis.text.x = element_text(angle=90)) + 
  xlab("Year of Release") + 
  ylab('Average rating of all movies released in the year')

# Check if some years of release have very less ratings requiring regularisation
edx %>% left_join(release_year_per_movie,by='movieId') %>% dplyr::count(year_of_release) %>% ggplot(aes(log10(n))) + geom_histogram(binwidth=0.15) + xlab('Number of ratings per year of release [log10(n) scale]')

# Compute lambda for regularization for year of release effect
lambdas <- seq(0,10,1)

lambda_to_use_per_partition <- sapply(1:5, function(index) {
  lambda_data_set <- prepare_lambda_data_set(index)
  train_lambda_set <- lambda_data_set[[1]]
  test_lambda_set <- lambda_data_set[[2]]
  
  lambda_rmses <- sapply(lambdas, function(lambda) {
    release_year_effects <- train_lambda_set %>% left_join(movie_effects,by='movieId') %>% left_join(user_effects,by='userId') %>% left_join(genre_effects,by='genres') %>% left_join(release_year_per_movie,by='movieId') %>% group_by(year_of_release) %>%  summarise(n=n(),b_y_hat=sum(rating - total_mean-b_i_hat-b_u_hat-b_g_hat)/(n+lambda))
    
    predicted_ratings <- test_lambda_set %>% left_join(release_year_per_movie,by='movieId') %>% left_join(movie_effects,by='movieId') %>% left_join(user_effects,by='userId') %>% left_join(genre_effects,by='genres') %>% left_join(release_year_effects,by='year_of_release') %>% mutate(pred=total_mean+b_i_hat+b_u_hat+b_g_hat+b_y_hat) %>% pull(pred)
    
    RMSE(test_lambda_set$rating,predicted_ratings) 
  })
  
  # selecting lambda leading to lowest RMSE
  lambda_to_use <- lambdas[which.min(lambda_rmses)]
  paste('Lambda to use: ', lambda_to_use, ', corresponding RMSE: ', lambda_rmses[which.min(lambda_rmses)])
  
  lambda_to_use
})

lambda_to_use_per_partition

# selecting lambda 
lambda_to_use <- mean(lambda_to_use_per_partition)
lambdas_per_effect <- lambdas_per_effect %>% add_row(effect_name='year_of_release',lambda=lambda_to_use)

rm(lambdas,lambda_to_use_per_partition)

# Post regularization for year of release effect
# So let's determine the release year effect
# y_u_i_hat = true_rating + b_i_hat + b_u_hat + b_g_hat + b_y_hat

# computed release year effect to be used later based on all paritions
release_year_effects_list <- lapply(1:5, function(index) {
  test_indices <- test_indices_list[[index]]
  train_set <- edx[-test_indices,]
  
  y_train_set <- train_set$rating
  
  release_year_effects <- train_set %>% left_join(movie_effects,by='movieId') %>% left_join(user_effects,by='userId') %>% left_join(genre_effects,by='genres') %>% left_join(release_year_per_movie,by='movieId') %>% group_by(year_of_release) %>%  summarise(n=n(),b_y_hat=sum(rating - total_mean-b_i_hat-b_u_hat-b_g_hat)/(n+lambda_to_use))
  
  
  release_year_effects
}
)

# as different partitions may not have exactly same data set combined the results
rye <- release_year_effects_list[[1]]
for (index in 2:5) {
  rye <- rye %>% full_join(release_year_effects_list[[index]],by='year_of_release', suffix=c("1",index))
}

rye <- rye %>% select(-n1,-n2,-n11,-n4,-n)
release_year_effects <- tibble(rye[,1], b_y_hat=rowMeans(rye[,-1],na.rm = TRUE))

# to ensure we don't get random rmse due to data set. Rmse calculated based on all partitions
rmses <- sapply(1:5, function(index) {
  test_indices <- test_indices_list[[index]]
  train_set <- edx[-test_indices,]
  test_set <- edx[test_indices,]
  
  x_train_set <- train_set %>% select(userId,movieId,genres)
  
  # Before we use test_set, it needs to be filtered as it could have rows which have movies, users not present in training data.
  # So, we use only that test data which has movieId, userId in the training data
  test_set <- test_set %>% semi_join(x_train_set,by="movieId") %>% semi_join(x_train_set,by="userId")
  
  x_test_set <- test_set %>% select(userId,movieId,genres)
  y_test_set <- test_set$rating
  
  release_year_effects <- train_set %>% left_join(movie_effects,by='movieId') %>% left_join(user_effects,by='userId') %>% left_join(genre_effects,by='genres') %>% left_join(release_year_per_movie,by='movieId') %>% group_by(year_of_release) %>%  summarise(n=n(),b_y_hat=sum(rating - total_mean-b_i_hat-b_u_hat-b_g_hat)/(n+lambda_to_use))
  
  
  predicted_ratings <- test_set %>% left_join(release_year_per_movie,by='movieId') %>% left_join(movie_effects,by='movieId') %>% left_join(user_effects,by='userId') %>% left_join(genre_effects,by='genres') %>% left_join(release_year_effects,by='year_of_release') %>% mutate(pred=total_mean+b_i_hat+b_u_hat+b_g_hat+b_y_hat) %>% pull(pred)
  
  RMSE(y_test_set,predicted_ratings)
}
)
# Let's see how the release year effect varies
release_year_effects %>% ggplot(aes(b_y_hat)) + geom_histogram(bins = 100,color=I('black'))

# Let's evaluate performance for this model
# y_u_i_hat = true_rating + b_i_hat + b_u_hat + b_g_hat + b_y_hat
performance_metric <- performance_metric %>% add_row(model_name='With movie, user, genre and release year effect model',rmse=mean(rmses))

performance_metric %>% knitr::kable()
rm(index, lambda_to_use, release_year, rmses, movies,rye,release_year_effects_list)


#######################################################################################
#################### Final prepared model training and evaluation #####################
#######################################################################################
# Post evaluation with multiple predictors, the final model resulted to the below form:
# y_u_i_hat = true_rating + b_i_hat + b_u_hat + b_g_hat + b_y_hat

# since the model was now identified and regularized, let's train the final model over the entire training data set i.e. edx and then test it against the validation data set i.e. validation.

# Training model
# Training model
set.seed(1, sample.kind = 'Rounding')

lambda_to_use <- lambdas_per_effect %>% filter(effect_name=='movies') %>% pull(lambda)
movie_effects <- edx %>% group_by(movieId) %>% summarise(n=n(),b_i_hat=sum(rating - total_mean)/(n+lambda_to_use));

lambda_to_use <- lambdas_per_effect %>% filter(effect_name=='users') %>% pull(lambda)
user_effects <- edx %>% left_join(movie_effects,by='movieId') %>% group_by(userId) %>% summarise(n=n(),b_u_hat=sum(rating - total_mean - b_i_hat)/(n+lambda_to_use))

lambda_to_use <- lambdas_per_effect %>% filter(effect_name=='genres') %>% pull(lambda)
genre_effects <- edx %>% left_join(movie_effects,by='movieId') %>% left_join(user_effects,by='userId') %>% group_by(genres) %>% summarise(n=n(),b_g_hat=sum(rating - total_mean-b_i_hat-b_u_hat)/(n+lambda_to_use))

lambda_to_use <- lambdas_per_effect %>% filter(effect_name=='year_of_release') %>% pull(lambda)
release_year_effects <- edx %>% left_join(movie_effects,by='movieId') %>% left_join(user_effects,by='userId') %>% left_join(genre_effects,by='genres') %>% left_join(release_year_per_movie,by='movieId') %>% group_by(year_of_release) %>%  summarise(n=n(),b_y_hat=sum(rating - total_mean-b_i_hat-b_u_hat-b_g_hat)/(n+lambda_to_use))

# Validating model
# predictons 
predicted_ratings <- validation %>% left_join(release_year_per_movie,by='movieId') %>% left_join(movie_effects,by='movieId') %>% left_join(user_effects,by='userId') %>% left_join(genre_effects,by='genres') %>% left_join(release_year_effects,by='year_of_release') %>% mutate(pred=total_mean+b_i_hat+b_u_hat+b_g_hat+b_y_hat) %>% pull(pred)

# Let's evaluate performance for this model
# y_u_i_hat = true_rating + b_i_hat + b_u_hat + b_g_hat + b_y_hat
performance_metric <- performance_metric %>% add_row(model_name='Final model',rmse=RMSE(validation$rating,predicted_ratings))

performance_metric %>% knitr::kable()
