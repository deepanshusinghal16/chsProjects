library(ggplot2)
# Load the mtcars dataset
data(mtcars)

# Perform linear regression
linear_regression <- lm(mpg ~ hp, data = mtcars)

# Display the summary of the linear regression model
summary(linear_regression)

# Visualize the results using ggplot2
library(ggplot2)
ggplot(data = mtcars, aes(x = hp, y = mpg)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE, color = "blue") +
  labs(title = "Linear Regression - Miles per Gallon vs. Horsepower",
       x = "Horsepower",
       y = "Miles per Gallon")