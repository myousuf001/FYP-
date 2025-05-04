library(readxl)
library(ggplot2)
library(GGally)
library(caret)
library(dplyr)
library(glmnet)


# Load the dataset (update file path)
df <- read_excel("/Users/yousufhabib/Downloads/bank-full.xls")

# View the first few rows
head(df)
# Check for missing values in each column
colSums(is.na(df))
# Check structure of the data
str(df)
# Convert categorical variables to factors
df$job <- as.factor(df$job)
df$marital <- as.factor(df$marital)
df$education <- as.factor(df$education)
df$default <- as.factor(df$default)
df$housing <- as.factor(df$housing)
df$loan <- as.factor(df$loan)
df$contact <- as.factor(df$contact)
df$month <- as.factor(df$month)
df$poutcome <- as.factor(df$poutcome)
df$y <- as.factor(df$y)  # Target variable

# Summary statistics for numerical variables
summary(df)

# Count of each category in target variable
df <- df %>%
  mutate(across(where(is.character), as.factor))

# Ensure the target variable (y) is a factor with "yes" as the positive class
df$y <- factor(df$y, levels = c("no", "yes"))

# Set seed for reproducibility
set.seed(123)

# Split the data into 80% training and 20% testing