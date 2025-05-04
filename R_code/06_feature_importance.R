rf_importance_df <- data.frame(Feature = rownames(rf_importance), Importance = rf_importance[, 1])

# Remove NA values if any
rf_importance_df <- rf_importance_df[!is.na(rf_importance_df$Feature), ]

# Plot Random Forest Feature Importance
ggplot(rf_importance_df, aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "red") +
  coord_flip() +
  theme_minimal() +
  ggtitle("Top Features - Random Forest")
