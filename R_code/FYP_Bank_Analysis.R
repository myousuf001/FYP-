# Install and load necessary libraries
install.packages("readxl")  # Install if not already installed
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
table(df$y)



# Histogram of Age
ggplot(df, aes(x = age)) +
  geom_histogram(fill = "blue", color = "black", bins = 30) +
  theme_minimal() +
  ggtitle("Distribution of Age")

# Boxplot of Balance by Subscription Outcome
ggplot(df, aes(x = y, y = balance, fill = y)) +
  geom_boxplot() +
  theme_minimal() +
  ggtitle("Balance vs Subscription Outcome")

# Bar plot for Job categories
ggplot(df, aes(x = job, fill = y)) +
  geom_bar(position = "fill") +  
  coord_flip() +  
  theme_minimal() +
  ggtitle("Subscription Rate by Job Type")

# Bar plot for Marital Status
ggplot(df, aes(x = marital, fill = y)) +
  geom_bar(position = "fill") +  
  theme_minimal() +
  ggtitle("Subscription Rate by Marital Status")




# Correlation matrix
ggcorr(df[, sapply(df, is.numeric)], label = TRUE)


# Convert categorical variables to factors
df <- df %>%
  mutate(across(where(is.character), as.factor))

# Ensure the target variable (y) is a factor with "yes" as the positive class
df$y <- factor(df$y, levels = c("no", "yes"))

# Set seed for reproducibility
set.seed(123)

# Split the data into 80% training and 20% testing
train_index <- createDataPartition(df$y, p = 0.8, list = FALSE)
train_data <- df[train_index, ]
test_data <- df[-train_index, ]

# Check data split
train_data$y <- ifelse(train_data$y == "yes", 1, 0)
test_data$y  <- ifelse(test_data$y == "yes", 1, 0)

# Confirm conversion
table(train_data$y)
table(test_data$y)

# Identify problematic variables (with only 1 unique level)
problematic_vars <- sapply(train_data, function(x) if(is.factor(x)) length(unique(x)) else NA)
problematic_vars <- names(problematic_vars[problematic_vars == 1])
print(problematic_vars)  # Shows which columns are causing issues

# Remove these problematic variables
train_data <- train_data %>% select(-poutcome)
test_data <- test_data %>% select(-poutcome)
# Convert categorical variables to factors again
train_data <- train_data %>% mutate(across(where(is.character), as.factor))
test_data <- test_data %>% mutate(across(where(is.character), as.factor))

# Ensure target variable (y) is numeric (for regression)
train_data$y <- as.numeric(as.character(train_data$y))
test_data$y <- as.numeric(as.character(test_data$y))

# Check if the issue is fixed
str(train_data)

#Implementing logisitc regression model
train_data <- train_data %>% select(-pdays, -previous)
test_data <- test_data %>% select(-pdays, -previous)
logistic_model <- glm(y ~ ., data = train_data, family = binomial)
summary(logistic_model)

# Predict on test data
test_data$predicted_probs <- predict(logistic_model, newdata = test_data, type = "response")
test_data$predicted_class <- ifelse(test_data$predicted_probs > 0.5, 1, 0)

# Confusion Matrix
conf_matrix <- table(Predicted = test_data$predicted_class, Actual = test_data$y)
print(conf_matrix)

# Accuracy Calculation
accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)
print(paste("Accuracy:", round(accuracy, 4)))

# ROC Curve and AUC Score
library(pROC)
roc_curve <- roc(test_data$y, test_data$predicted_probs)
plot(roc_curve, main="ROC Curve for Logistic Regression", col="blue")
auc_score <- auc(roc_curve)
print(paste("AUC Score:", round(auc_score, 4)))

# Fine-tuning logisitc regression model

set.seed(123)
cv_model <- cv.glmnet(train_x, train_y, alpha = 0.5, family = "binomial")  # Elastic Net
best_lambda <- cv_model$lambda.1se  # Optimal Lambda
print(paste("Best Lambda:", best_lambda))

logistic_tuned <- glmnet(train_x, train_y, alpha = 0.5, family = "binomial", lambda = best_lambda)
print(logistic_tuned)

# Make predictions
pred_probs <- predict(logistic_tuned, newx = test_x, type = "response")
pred_classes <- ifelse(pred_probs > 0.5, 1, 0)

# Confusion Matrix
conf_matrix <- table(Predicted = pred_classes, Actual = test_y)
print(conf_matrix)

# Accuracy Calculation
accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)
print(paste("Accuracy:", round(accuracy, 4)))

# Compute ROC Curve
pred_probs <- as.numeric(pred_probs)  

roc_curve <- roc(test_y, pred_probs)
plot(roc_curve, main="ROC Curve for Fine-Tuned Logistic Regression", col="blue")

# Compute AUC Score
auc_score <- auc(roc_curve)
print(paste("AUC Score:", round(auc_score, 4)))

#Implement Random Forest Model

library(randomForest)
# Ensure target variable is categorical
train_data$y <- as.factor(train_data$y)
test_data$y <- as.factor(test_data$y)

# Verify structure
str(train_data$y)

# Set seed for reproducibility
set.seed(123)

# Tune mtry to find the best value
best_mtry <- tuneRF(train_data[-ncol(train_data)], train_data$y, stepFactor = 1.5, improve = 0.01, trace = TRUE, plot = TRUE)
print(best_mtry)

# Train Random Forest Model
rf_model <- randomForest(y ~ ., data = train_data, ntree = 100, mtry = 3, importance = TRUE)

# Print model summary
print(rf_model)

# Make predictions on the test set
predictions <- predict(rf_model, test_data, type = "class")

# Confusion Matrix
conf_matrix <- table(Predicted = predictions, Actual = test_data$y)
print(conf_matrix)

# Accuracy Calculation
accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)
print(paste("Accuracy:", round(accuracy, 4)))

# Get prediction probabilities for AUC calculation
pred_probs <- predict(rf_model, test_data, type = "prob")[,2]

# Compute ROC curve
roc_curve <- roc(test_data$y, pred_probs)

# Plot ROC curve
plot(roc_curve, main="ROC Curve for Random Forest", col="blue")

# Calculate AUC Score
auc_score <- auc(roc_curve)
print(paste("AUC Score:", round(auc_score, 4)))


# Fine tuning Random Forest Model

# Set seed for reproducibility
set.seed(123)

# Train the fine-tuned Random Forest Model with mtry = 3
rf_tuned <- randomForest(y ~ ., data=train_data, ntree=300, mtry=3, importance=TRUE)

# Print model summary
print(rf_tuned)

# Make Predictions
pred_probs_rf <- predict(rf_tuned, test_data, type="prob")[,2]
pred_classes_rf <- predict(rf_tuned, test_data, type="class")

# Compute Confusion Matrix
conf_matrix_rf <- table(Predicted=pred_classes_rf, Actual=test_data$y)
print(conf_matrix_rf)

# Calculate Accuracy
accuracy_rf <- sum(diag(conf_matrix_rf)) / sum(conf_matrix_rf)
print(paste("Accuracy:", round(accuracy_rf, 4)))

# Compute ROC Curve and AUC Score
roc_curve_rf <- roc(test_data$y, as.numeric(pred_probs_rf))
plot(roc_curve_rf, main="ROC Curve for Fine-Tuned Random Forest", col="blue")

# Compute AUC Score
auc_score_rf <- auc(roc_curve_rf)
print(paste("AUC Score:", round(auc_score_rf, 4)))


# ------------------------------
# FEATURE IMPORTANCE ANALYSIS
# ------------------------------

# 📌 1. Feature Importance from Logistic Regression (Fine-Tuned)
coefficients <- coef(logistic_tuned)[-1]  # Remove intercept
feature_names <- rownames(coef(logistic_tuned))[-1]  # Remove intercept row

# Convert to Data Frame
feature_importance_logistic <- data.frame(Feature = feature_names, Coefficient = coefficients)

# Plot Logistic Regression Feature Importance
library(ggplot2)
ggplot(feature_importance_logistic, aes(x = reorder(Feature, Coefficient), y = Coefficient)) +
  geom_bar(stat = "identity", fill = "blue") +
  coord_flip() +
  theme_minimal() +
  ggtitle("Top Features - Logistic Regression")




# 📌 2. Feature Importance from Random Forest (Fine-Tuned)
rf_importance <- importance(rf_tuned)  # Extract feature importance
rf_importance_df <- data.frame(Feature = rownames(rf_importance), Importance = rf_importance[, 1])

# Remove NA values if any
rf_importance_df <- rf_importance_df[!is.na(rf_importance_df$Feature), ]

# Plot Random Forest Feature Importance
ggplot(rf_importance_df, aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "red") +
  coord_flip() +
  theme_minimal() +
  ggtitle("Top Features - Random Forest")

