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
conf_matrix <- table(Predicted = pred_classes, Actual = test_y)
print(conf_matrix)

# Accuracy Calculation
accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)
print(paste("Accuracy:", round(accuracy, 4)))

# Compute ROC Curve
pred_probs <- as.numeric(pred_probs)  

roc_curve <- roc(test_y, pred_probs)
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