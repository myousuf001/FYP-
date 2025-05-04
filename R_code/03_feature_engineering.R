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