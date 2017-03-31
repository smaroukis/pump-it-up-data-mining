# Define train_values_url
train_values_url <- "http://s3.amazonaws.com/drivendata/data/7/public/4910797b-ee55-40a7-8668-10efd5c1b960.csv"

# Import train_values
train_values <- read.csv(train_values_url)

# Define train_labels_url
train_labels_url <- "http://s3.amazonaws.com/drivendata/data/7/public/0bf8bc6e-30d0-4c50-956a-603fc693d966.csv"

# Import train_labels
train_labels <- read.csv(train_labels_url)

# Define test_values_url
test_values_url <- "http://s3.amazonaws.com/drivendata/data/7/public/702ddfc5-68cd-4d1d-a0de-f5f566f76d91.csv"

# Import test_values
test_values <- read.csv(test_values_url)

# Merge data frames to create the data frame train
train <- merge(train_labels, train_values)

# Look at the number of pumps in each functional status group
table(train$status_group)

library(ggplot2)
str(train)

library(ggplot2)
library(googleVis)

# Create scatter plot: latitude vs longitude with color as status_group
ggplot(subset(train[1:1000,], latitude < 0 & longitude > 0),
    aes(x = latitude, y = longitude, color = status_group)) + 
    geom_point(shape = 1) + 
    theme(legend.position = "top")

# Create a column 'latlong' to input into gvisGeoChart
train$latlong <- paste(round(train$latitude,2), round(train$longitude, 2), sep = ":")

# Use gvisGeoChart to create an interactive map with well locations
wells_map <- gvisGeoChart(train[1:1000,], locationvar = "latlong", 
                          colorvar = "status_group", sizevar = "Size", 
                          options = list(region = "TZ"))

# Plot wells_map
wells_map
