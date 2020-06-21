################################
# Create edx set, validation set
################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding")
# if using R 3.5 or earlier, use `set.seed(1)` instead
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

#Seperate edx data into train and test set
test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.2, list = FALSE)
train_set <- edx[-test_index,] #train
temp2 <- edx[test_index,] #test

# Make sure userId and movieId in test set are also in train set
test_set <- temp2 %>% 
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId") 

# Add rows removed from test set back into train set
removed <- anti_join(temp2, test_set)
train_set <- rbind(train_set, removed)

#explore data set
head(edx)
dim(edx)


#plot show distribution of user and number of rating
edx %>% group_by(userId) %>%
  summarise(n=n()) %>%
  ggplot(aes(n)) +
  geom_histogram(color = "black") +
  scale_x_log10() + 
  ggtitle("Distribution of Users") +
  xlab("Number of Ratings") +
  ylab("Number of Users")

#plot show distrupution of number of rating between movies
edx %>% group_by(movieId) %>%
  summarise(n=n()) %>%
  ggplot(aes(n)) +
  geom_histogram(color="Black") +
  scale_x_log10()+
  ggtitle("Distribution of Movies") +
  xlab("Number of Ratings") +
  ylab("Number of Movies")

#plot show distribution of rating among movies
edx %>% group_by(movieId) %>%
  summarise(avg_rating=mean(rating)) %>%
  ggplot(aes(avg_rating)) +
  geom_histogram(color="Black")

#plot show variations among users
train_set %>% group_by(userId) %>% mutate(b_u=mean(rating)) %>% 
  ggplot(aes(b_u)) + 
  geom_histogram(bins = 30,color="Black")

# Overall rating for all movie across all user
mu_hat <- mean(train_set$rating)
RMSE(mu_hat,test_set$rating)

#Movie effect
movie_avg <- train_set %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating-mu_hat))
RMSE(test_set %>% 
       left_join(movie_avg, by = "movieId") %>%
       mutate(pred = mu_hat + b_i) %>%
       pull(pred),test_set$rating)

#user effect
user_avg <- train_set %>%
  left_join(movie_avg,by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u=mean(rating-mu_hat-b_i))
RMSE(test_set %>% 
       left_join(movie_avg, by = "movieId") %>%
       left_join(user_avg, by = "userId") %>%
       mutate(pred = mu_hat + b_i + b_u) %>%
       pull(pred),test_set$rating)
#Extreme prediction movies tend to have low number of user
#Top 5 best prediction rating movies
train_set %>% dplyr::count(movieId) %>% 
  left_join(movie_avg) %>%
  left_join(train_set %>% 
              select(movieId, title) %>%
              distinct(), by="movieId") %>%
  arrange(desc(b_i)) %>% 
  select(title, b_i, n) %>% 
  slice(1:5) %>% 
  knitr::kable()
#Top 5 worst prediction rating movies
train_set %>% dplyr::count(movieId) %>% 
  left_join(movie_avg) %>%
  left_join(train_set %>% 
              select(movieId, title) %>%
              distinct(), by="movieId") %>%
  arrange(b_i) %>% 
  select(title, b_i, n) %>% 
  slice(1:5) %>% 
  knitr::kable()

#Cross validation to choose lampda for regularization
lamdas <- seq(0,10,0.25)
rmses <- sapply(lamdas,function(l){
  
  mu <- mean(train_set$rating)
  
  b_i <- train_set %>% group_by(movieId) %>%
    summarize(b_i=sum(rating-mu)/(n()+l))
  
  b_u <- train_set %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  
  predicted_ratings <- test_set %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    pull(pred)
  
  return(RMSE(predicted_ratings, test_set$rating))
})
qplot(lamdas,rmses)
lamdas[which.min(rmses)]

#Apply lampda that has lowest rmse
b_i <- train_set %>%
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu_hat)/(n()+lamdas[which.min(rmses)]))
b_u <- train_set %>% 
  left_join(b_i, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu_hat)/(n()+lamdas[which.min(rmses)]))
predicted_ratings <- 
  test_set %>% 
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  mutate(pred = mu_hat + b_i + b_u) %>%
  .$pred
RMSE(predicted_ratings, test_set$rating)

#Matrix factorization
library(recosystem)

#convert train and test set to input of recosystem
reco_train <- with(train_set, data_memory(user_index = userId, 
                                    item_index = movieId, 
                                    rating = rating))
reco_test <- with(test_set, data_memory(user_index = userId,
                                        item_index = movieId,
                                        rating = rating))

#create model object
r_train <- Reco()

#tune pararmeter by cross validation
opts_train <- r_train$tune(reco_train,opts = list(dim      = c(10L, 20L, 30L, 40L),
                                    costp_l1 = c(0, 0.1),
                                    costp_l2 = c(0.01, 0.1),
                                    costq_l1 = c(0, 0.1),
                                    costq_l2 = c(0.01, 0.1),
                                    lrate    = c(0.01, 0.1, 0.2),
                                    nthread  = 4,
                                    niter = 10)
)

#train model using best tuning parameter
r_train$train(reco_train, opts = c(opts_train$min, nthread = 4, niter = 20))

#make prediction for validation set
y_hat_train <-  r_train$predict(reco_test, out_memory())

#calculate rmse
RMSE(test_set$rating, y_hat_train)


#convert edx to input of recosystem
reco_edx <-  with(edx, data_memory(user_index = userId, 
                                   item_index = movieId, 
                                   rating = rating))

#create model object
r <- Reco()

#tune pararmeter by cross validation
opts <- r$tune(reco_edx,opts = list(dim      = c(10L, 20L, 30L, 40L),
                                    costp_l1 = c(0, 0.1),
                                    costp_l2 = c(0.01, 0.1),
                                    costq_l1 = c(0, 0.1),
                                    costq_l2 = c(0.01, 0.1),
                                    lrate    = c(0.01, 0.1, 0.2),
                                    nthread  = 4,
                                    niter = 10)
)

#train model using best tuning parameter
r$train(reco_edx, opts = c(opts$min, nthread = 4, niter = 20))

#make prediction for validation set
reco_val  <-  with(validation, data_memory(user_index = userId, 
                                           item_index = movieId, 
                                           rating = rating))
y_hat <-  r$predict(reco_val, out_memory())

#calculate rmse
RMSE(validation$rating, y_hat)

data.frame(method=c("Guessing"),RMSE=c(RMSE(mu_hat,test_set$rating)))

       