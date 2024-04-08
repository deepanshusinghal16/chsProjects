# Load the required library
library(MASS)
library(ggplot2)

# Load the mtcars dataset
data(mtcars)

# Convert mpg to a factor variable
mtcars$mpg <- as.factor(mtcars$mpg)

# Define the model formula
formula <- vs ~ hp + wt

# Fit the logistic regression model
model <- glm(formula, data = mtcars, family = binomial)

# Display the model summary
summary(model)

# Load the required library


# Create a scatter plot with the regression line
ggplot(data = mtcars, aes(x = hp, y = as.numeric(mpg))) +
  geom_point() +
  geom_smooth(method = "glm", se = FALSE, method.args = list(family = "binomial")) +
  labs(x = "Horsepower", y = "MPG", title = "Logistic Regression Model")
