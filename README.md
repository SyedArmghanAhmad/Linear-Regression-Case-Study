
# California Housing Price Prediction with Linear Regression

This project investigates the feasibility of predicting California housing prices using a linear regression model with polynomial features. The project leverages the following key steps:

1. **Data Acquisition and Cleaning:**
   - Acquired the California housing price dataset.
   - Performed data cleaning tasks to address missing values and outliers.
   - Analyzed data distribution and explored potential relationships between features.

2. **Feature Engineering:**
   - Handled categorical features by creating dummy variables.
   - Dropped irrelevant features based on domain knowledge and correlation analysis.

3. **Model Training and Evaluation (Linear Regression):**
   - Split the data into training and testing sets.
   - Trained a linear regression model using the training data.
   - Evaluated the model's performance on the testing data using metrics like Mean Squared Error (MSE) and Root Mean Squared Error (RMSE).
   - Assessed the model's assumptions (linearity, homoscedasticity, etc.) through visualizations.

4. **Model Training and Evaluation (Polynomial Regression):**
   - Implemented polynomial features to capture non-linear relationships.
   - Trained a linear regression model using the polynomial features.
   - Compared the performance of the polynomial regression model with the base linear regression model using metrics like R-squared, MSE, and RMSE.

5. **Feature Importance Analysis:**
   - Identified the most important features for the polynomial regression model.
   - Visualized feature importance to understand the relative contribution of features to the model's predictions.

6. **Residual Analysis:**
   - Examined the distribution of residuals to assess model assumptions and identify potential areas for improvement.

7. **Visualization:**
   - Created various visualizations (histograms, boxplots, scatterplots) to explore data distribution, identify outliers, and understand relationships between features.

**Key Findings:**

- The linear regression model with polynomial features achieved a **Train R-squared of 0.6966** and a **Test R-squared of 0.7006**, indicating a good fit to the data.
- The model's performance on the testing data resulted in a Mean Squared Error (MSE) of **2402131680.745** and a Root Mean Squared Error (RMSE) of **49011.546**, suggesting relatively accurate predictions.
- Polynomial features yielded a slight improvement in model performance compared to the base linear regression model.
- Feature importance analysis provided insights into the most influential factors affecting California housing prices.

**Future Considerations:**

- Explore alternative regression models (e.g., Ridge, Lasso) to potentially improve model generalizability and address overfitting.
- Investigate the impact of hyperparameter tuning on model performance.
- Incorporate additional features or feature engineering techniques to potentially enhance model accuracy.
- Consider more sophisticated techniques for handling outliers and missing values.

**Overall, this project demonstrates the potential of linear regression with polynomial features for predicting California housing prices. The project provides a solid foundation for further exploration and refinement of the model.**
