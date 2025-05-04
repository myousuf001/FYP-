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