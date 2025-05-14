plot(roc_curve, main="ROC Curve for Logistic Regression", col="blue")
auc_score <- auc(roc_curve)
print(paste("AUC Score:", round(auc_score, 4)))

# Fine-tuning logisitc regression model
# Create model matrices for glmnet (excluding intercept)
train_x <- model.matrix(y ~ . -1, data = train_data)
test_x <- model.matrix(y ~ . -1, data = test_data)

# Target variables
train_y <- train_data$y
test_y <- test_data$y

set.seed(123)
# Create model matrices for glmnet (excluding intercept)
train_x <- model.matrix(y ~ . -1, data = train_data)
test_x <- model.matrix(y ~ . -1, data = test_data)

# Target variables
train_y <- train_data$y
test_y <- test_data$y
cv_model <- cv.glmnet(train_x, train_y, alpha = 0.5, family = "binomial")  # Elastic Net
best_lambda <- cv_model$lambda.1se  # Optimal Lambda
print(paste("Best Lambda:", best_lambda))

logistic_tuned <- glmnet(train_x, train_y, alpha = 0.5, family = "binomial", lambda = best_lambda)
print(logistic_tuned)

# Make predictions
pred_probs <- predict(logistic_tuned, newx = test_x, type = "response")
pred_classes <- ifelse(pred_probs > 0.5, 1, 0)

# Confusion Matrix
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
plot(roc_curve, main="ROC Curve for Random Forest", col="blue")

# Calculate AUC Score
auc_score <- auc(roc_curve)
print(paste("AUC Score:", round(auc_score, 4)))


# 📌 Fine-Tuning Random Forest Model

# Set seed for reproducibility
set.seed(123)

# Train the fine-tuned Random Forest Model with mtry = 3
rf_tuned <- randomForest(y ~ ., data = train_data, ntree = 300, mtry = 3, importance = TRUE)

# Print model summary
print(rf_tuned)

# Make Probability Predictions
pred_probs_rf <- predict(rf_tuned, test_data, type = "prob")[,2]

# Adjusted threshold (from 0.5 to 0.25) to increase recall
pred_classes_rf <- ifelse(pred_probs_rf > 0.25, 1, 0)

# Confusion Matrix
conf_matrix_rf <- table(Predicted = pred_classes_rf, Actual = test_data$y)
print(conf_matrix_rf)

# Calculate Recall
TP <- conf_matrix_rf["1", "1"]
FN <- conf_matrix_rf["0", "1"]
recall <- TP / (TP + FN)
print(paste("Recall:", round(recall, 4)))

# Calculate Accuracy
accuracy_rf <- sum(diag(conf_matrix_rf)) / sum(conf_matrix_rf)
print(paste("Accuracy:", round(accuracy_rf, 4)))

# Compute ROC Curve and AUC Score
roc_curve_rf <- roc(test_data$y, pred_probs_rf)
plot(roc_curve_rf, main = "ROC Curve for Fine-Tuned Random Forest (Threshold 0.25)", col = "blue")

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
