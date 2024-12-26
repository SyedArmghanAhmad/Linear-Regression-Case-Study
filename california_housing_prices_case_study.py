import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


# # Loading California House Pricing Dataset
#Creating a filepath for dataset
file_path =r"E:\Machine Learning Projects\Linear Regression Case Study\datasets\housing.csv"
output = r"E:\Machine Learning Projects\Linear Regression Case Study\Visualizations"
#Reading the Dataset
data = pd.read_csv(file_path)
#Displaying it
print('\n')
print("--- Displaying Dataset ---\n")
print(data.head())
print('\n')
print("--- Displaying Columns ---\n")
print(data.columns)
print('\n')
# displaying first 10 rows
print("--- Displaying First 10 rows ---\n")
print(data.head(10))

#Data Exploration
print('\n')
print("--- Data Exploration ---\n")
print("Data Types:\n")
print(data.dtypes)
print("\nNumber of Rows and Columns:\n")
print(data.shape)
print("--- Data Count ---\n")
print(data.info())

# # Missing Data Values
#check for missing values
missing_values = data.isnull().sum()

#Checking the percentage of missing values in each column
missing_percentage = (missing_values / len(data))*100

#displaying the Calculations
print('\n')
print("Missing values in Each Column:\n", missing_values)
print('\n')
print("\nPercentage of Missing Data:\n", missing_percentage)
print('\n')
#Remove Rows with Missing values
data_cleaned = data.dropna()
#Verify that Missing Values have been Removed
print('\n')
print("Missing Values in Each column after removal:\n")
print(data_cleaned.isnull().sum())

# # Data Exploration and Visualization
#Statistical description of data
print('\n')
print("Statistical description of data:\n")
print(data.describe())

# Visualization

sns.set_theme(style = "whitegrid")
plt.figure(figsize = (10,6))
sns.histplot(data_cleaned['median_house_value'], color = 'forestgreen' , kde=True)
plt.title('Distribution of Median House Value')
plt.xlabel('Median House Value')
plt.ylabel('Frequency')
plt.savefig(f"{output}/Distribution of Median House Value.jpg",format='jpg', dpi=300 ,bbox_inches='tight')
plt.show()

#Assuming 'data' is your DataFrame and 'median_house_value' is column of interest
print('\n')
print("--- Handling Outliers ---\n")

Q1 = data_cleaned['median_house_value'].quantile(0.25)
print('25% percentile: ')
print(Q1)
Q3 = data_cleaned['median_house_value'].quantile(0.75)
print('75% percentile: ')
print(Q3)
IQR = Q3 - Q1
print('Inter Quartile Range: ')
print(IQR)


#Define the bound for the outliers
lower_bound = Q1 - 1.5 *  IQR
upper_bound = Q3 + 1.5 *  IQR

#Remove outliers
data_outliers_1 = data_cleaned[(data_cleaned['median_house_value'] >= lower_bound) & (data_cleaned['median_house_value'] <= upper_bound)]

#Check the shape of the data before and after removal of outliers
print("Shape of data before removing outliers:", data.shape)
print("Shape of data after removing outliers:", data_cleaned.shape)

# BoxPlot for Outlier Detection
## Outliers in Median Income
###Checking boxplot for outliers detection
plt.figure(figsize=(10,6))
sns.boxplot(x=data_outliers_1['median_income'], color='purple')
plt.title('Outlier Analysis of Median Income')
plt.xlabel('Median Income')

#Save the boxplot to a file
plt.savefig(f"{output}/Outlier Analysis of Median Income.jpg",format='jpg', dpi=300, bbox_inches='tight')
plt.show()

#Calculating Q1 and Q3
print('\n')
print("--- Handling Outliers Again ---\n")
Q1 = data_outliers_1['median_house_value'].quantile(0.25)
print('25% percentile: ')
print(Q1)
Q3 = data_outliers_1['median_house_value'].quantile(0.75)
print('75% percentile: ')
print(Q3)
IQR = Q3 - Q1
print('Inter Quartile Range: ')
print(IQR)

#Define the bound for the outliers
lower_bound = Q1 - 1.5 *  IQR
upper_bound = Q3 + 1.5 *  IQR

#Remove outliers
data_outliers_2 = data_outliers_1[(data_outliers_1['median_house_value'] >= lower_bound) & (data_outliers_1['median_house_value'] <= upper_bound)]

#Check the shape of the data before and after removal of outliers
print('\n')
print("Shape of data before removing outliers:", data_outliers_1.shape)
print("Shape of data after removing outliers:", data_outliers_2.shape)



#Again checking boxplot for outliers detection
plt.figure(figsize=(10,6))
sns.boxplot(x=data_outliers_2['median_income'], color='purple')
plt.title('Outlier Analysis of Median Income after Outlier Analysis Again')
plt.xlabel('Median Income')

#Save the boxplot to a file
plt.savefig(f"{output}/Outlier Analysis of Median Income after Outlier Analysis Again.jpg",format='jpg', dpi=300, bbox_inches='tight')
plt.show()

#saving the cleaned data to data
print('\n')
print("--- Saving Cleaned data to Data ---\n")
data = data_outliers_2


## Correlaton HeatMap
plt.figure(figsize=(10,6))
# Calculate the correlation matrix for numerical features only
numerical_data = data.select_dtypes(include=np.number)
sns.heatmap(numerical_data.corr(), annot=True, cmap='Greens')
plt.title('Correlation Heatmap')

plt.savefig(f"{output}/Correlation Heatmap.jpg",format='jpg', dpi=300, bbox_inches='tight')
plt.show()

#dropping the highest correlated independent variable total_bedrooms
print('\n')
print("--- Dropping the highest correlated independent variable total_bedrooms ---\n")
data = data.drop('total_bedrooms', axis=1)
print('\n')
print("--- Displaying columns again after dropping it ---\n")
print(data.columns)



# Here we have an independent variable named ocean_proximity which has string value:
# ['NEAR BAY' '<1H OCEAN' 'INLAND' 'NEAR OCEAN' 'ISLAND'].
#
# we need to tackle that before moving onto causal analysis with linear regression.
#
# There are multiple ways of handling this on the web
# but the better way is to transform it into string categorical variables...Dummy Variables.
#
# Dummy Variable Means that it takes two possible values
# 0 or 1 in this case Binary Values.
# 1 means the condition is Satisfied while 0 means condition is not satisfied.
#


#unique Value count foe categorical data
for column in ['ocean_proximity']:
  print(f"Unique Values in {column}: ", data[column].unique())

# String Data Categorization

## Check if 'ocean_proximity' exists before proceeding
if 'ocean_proximity' in data.columns:
  # Generate dummy variables
  ocean_proximity_dummies = pd.get_dummies(data['ocean_proximity'], prefix='ocean_proximity')
  data = pd.concat([data.drop("ocean_proximity", axis = 1), ocean_proximity_dummies], axis = 1).astype(int)
else:
  print("Column 'ocean_proximity' not found in the DataFrame.")

ocean_proximity_dummies.astype(int)

#checking columns
print('\n')
print("--- Checking Columns ---\n")
print(data.columns)
print("--- dropping the irrelevant columns from the dataset for further tuning ---\n")
#dropping the irrelevant columns from the dataset
data = data.drop('ocean_proximity_ISLAND', axis=1)
data = data.drop('total_rooms', axis =1)
print('\n')
print("--- Checking Columns After dropping them ---\n")
data.columns

#displaying the dataset again
print('\n')
print("--- Displaying Dataset ---\n")
print(data.head(10))


# Splitting the Test and Train data
##Define your Features[Independent Variables] and Target[Dependent Variable]
features = ['longitude', 'latitude', 'housing_median_age',
       'households', 'median_income','population',
       'ocean_proximity_<1H OCEAN', 'ocean_proximity_INLAND',
       'ocean_proximity_NEAR BAY', 'ocean_proximity_NEAR OCEAN']
target = ['median_house_value']

X = data[features]
y = data[target]

#Split the dat into testing and training set
#testing size determines the proportion of the data to be included in the test split
#random _state determines the reproducibility of your split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 1111)

#Checking the size of the split
print('\n')
print(f"Training test size: {X_train.shape[0]} samples")
print(f"Testing test size: {X_test.shape[0]} samples")


## Training
print('\n')
print("--- Training Start ---\n")
#Adding a constant to the predictor because statesmodel's OLS doesnt include it by defualt
X_train_const = sm.add_constant(X_train)

#fitting the model
model_fitted  = sm.OLS(y_train, X_train_const).fit()

#printing the Summary
print(model_fitted.summary())



# Prediction/Testing
print('\n')
print("--- Prediction/Testing ---\n")
##adding a constant to test predictors

X_test_const = sm.add_constant(X_test)

#making predictions
test_predictions = model_fitted.predict(X_test_const)
test_predictions

# Checking OLS Assumptions
print('\n')
print("--- Checking OLS Assumptions ---\n")
## Assumption#1: Linearty
print('\n')
print("--- Checking Assumption#1: Linearty ---\n")
###Scatter plot for observed vs predicted values on test data
plt.scatter(y_test, test_predictions, color='forestgreen')
plt.xlabel('Observed Values')
plt.ylabel('Predicted Values')
plt.title('Observed vs Predicted Values on Test Data')
plt.plot(y_test,y_test,color='darkred') #line for True Data

plt.savefig(f"{output}/Observed vs Predicted Values on Test Data-Checking Linearty.jpg",format='jpg', dpi=300, bbox_inches='tight')
plt.show()
print('\n')
print("--- # **Positive Linear Relationship:** the red line(which represents perfect prediction line) and the distribution of data-points suggest that there's positive linear relationship between observed and predicted values.This means if the Actual Values increase, the predicted ones will also increase. Which is a good sign for linearty ---\n")

# **Positive Linear Relationship:** the red line(which represents perfect prediction line) and the distribution of data-points suggest that there's positive linear relationship between observed and predicted values.This means if the Actual Values increase, the predicted ones will also increase.
# Which is a good sign for linearty

## Assumption#2 : Random Sample
print('\n')
print("--- Checking Assumption#2: Linearty ---\n")
mean_residuals = np.mean(model_fitted.resid)
print('\n')
print(f"the Mean of residuals is: {np.round(mean_residuals,2)}")

#PLotting the residuals
plt.scatter(model_fitted.fittedvalues, model_fitted.resid, color='forestgreen')
plt.axhline(y=0, color='darkred', linestyle='--')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted Values')

plt.savefig(f"{output}/Residuals vs Fitted Values-Checking Random Sample.jpg",format='jpg', dpi=300, bbox_inches='tight')
plt.show()
print('\n')
print("--- In this plot we dont see any dicsernible patterns. the residuals are thus randomly distributed among the horizontal line at 0 with no clear shape or trend.if there's a pattern or the residuals show a systematics deviation from zero then it could suggest issues such as model misspecifications, non-linearty or ommited-variable bias ---\n")

# in this plot we dont see any dicsernible patterns. the residuals are thus randomly distributed among the horizontal line at 0 with no clear shape or trend.
# if there's a pattern or the residuals show a systematics deviation from zero then it could suggest issues such as model misspecifications, non-linearty or ommited-variable bias

## Assumption 3: Exogeneity
print('\n')
print("--- Checking Assumption 3: Exogeneity ---\n")
#Calculate the residuals
residuals = model_fitted.resid

#check for correlation between residuals and predictor
for column in X_train.columns:
  corr_coefficent = np.corrcoef(X_train[column], residuals)[0,1]
  print(f"Correlation between residuals and {column} : {np.round(corr_coefficent,2)}")

print('\n')
print("--- Residuals values are 0.0 which is Good ---\n")


## Assumption 4 : Homoskedasticity
print('\n')
print("--- Checking Assumption 4 : Homoskedasticity ---\n")
#Plotting the Residuals
plt.scatter(model_fitted.fittedvalues, model_fitted.resid, color='forestgreen')
plt.axhline(y=0, color='darkred', linestyle='--')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted Values')

plt.savefig(f"{output}/Residuals vs Fitted Values-Checking Homoskedasticity.jpg",format='jpg', dpi=300, bbox_inches='tight')
plt.show()
print('\n')
print("--- **Random Scatter :** if the plot shows random scatter of residuals around the horizontal line at zero , it supports the homoescedasticity assumption. ---\n")
# **Random Scatter :** if the plot shows random scatter of residuals around the horizontal line at zero , it supports the homoescedasticity assumption.
## Training/Test/Evaluation with ScikitLearn
print('\n')
print("--- Training/Test/Evaluation with ScikitLearn ---\n")
### Scaling the data
#Initialize the Standard Scaler
scaler = StandardScaler()

#Fit and transform the training data
X_train_scaled = scaler.fit_transform(X_train)

#Transform the test data
X_test_scaled = scaler.transform(X_test)

#Create and fit the model
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)

#Make predictions on the test set
y_pred = lr.predict(X_test_scaled)

#Calculate MSE and RMSE
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print('\n')
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")

from sklearn.preprocessing import PolynomialFeatures

# Initialize the Standard Scaler and PolynomialFeatures
scaler = StandardScaler()
poly = PolynomialFeatures(degree=3, include_bias=False)

# Fit and transform the training data for scaling
X_train_scaled = scaler.fit_transform(X_train)

# Transform the test data for scaling
X_test_scaled = scaler.transform(X_test)

# Apply PolynomialFeatures to both the training and test data
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)

# Create and fit the model using Polynomial features
lr = LinearRegression()
lr.fit(X_train_poly, y_train)

# Make predictions on the test set
y_pred_poly = lr.predict(X_test_poly)

# Calculate MSE and RMSE
mse_poly = mean_squared_error(y_test, y_pred_poly)
rmse_poly = np.sqrt(mse_poly)

print(f"Polynomial Features Mean Squared Error (MSE): {mse_poly}")
print(f"Polynomial Features Root Mean Squared Error (RMSE): {rmse_poly}")

from sklearn.metrics import r2_score

# Calculate R-squared using the polynomial features for prediction
train_r2 = r2_score(y_train, lr.predict(X_train_poly)) # Changed X_train to X_train_poly
test_r2 = r2_score(y_test, lr.predict(X_test_poly)) # Changed X_test to X_test_poly
print('\n')
print(f"Train R-squared: {train_r2}")
print(f"Test R-squared: {test_r2}")


# 1. Feature Importance Plot (For Linear Regression coefficients)
## Get feature names after polynomial transformation
feature_names = poly.get_feature_names_out(X_train.columns)

feature_importance = pd.DataFrame({
    'Feature': feature_names,  # Use feature names from PolynomialFeatures
    'Importance': lr.coef_[0]  # Access the first element of coef_ for linear coefficients
})

# Sorting by importance
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

# Select top 20 important features (or change the number as needed)
top_n = 20
feature_importance = feature_importance.head(top_n)


# Plotting feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig(f"{output}/Feature Importance.jpg",format='jpg', dpi=300, bbox_inches='tight')
plt.show()

# 2. Residual Plot
# Calculate residuals
# Predict using polynomial features for train and test data
y_train_pred = lr.predict(X_train_poly)  # Changed X_train to X_train_poly
y_test_pred = lr.predict(X_test_poly)  # Changed X_test to X_test_poly

residuals_train = y_train - y_train_pred
residuals_test = y_test - y_test_pred

# Plotting residuals
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_train_pred.flatten(), y=residuals_train.values.flatten(), color='blue', alpha=0.6, label='Train')  # Flattened arrays
sns.scatterplot(x=y_test_pred.flatten(), y=residuals_test.values.flatten(), color='red', alpha=0.6, label='Test')  # Flattened arrays
plt.axhline(0, linestyle='--', color='black')
plt.title('Residual Plot')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.legend()
plt.tight_layout()
plt.savefig(f"{output}/Residual Plot.jpg", format='jpg', dpi=300, bbox_inches='tight')
plt.show()

# 3. Prediction vs Actual Plot
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_test_pred, color='green', alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2)
plt.title('Prediction vs Actual Values')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.tight_layout()
plt.savefig(f"{output}/Prediction vs Actual Values.jpg", format='jpg', dpi=300, bbox_inches='tight')
plt.show()

print('\n')
print("---========================Conclusion========================---\n")
print("This project successfully implemented a linear regression model with polynomial features to predict California housing prices. The model demonstrated promising results with a Train R-squared of 0.6966 and a Test R-squared of 0.7006, indicating a good fit to the data. The model's performance was further evaluated using Mean Squared Error (MSE) of 2402131680.745 and Root Mean Squared Error (RMSE) of 49011.546, suggesting relatively accurate predictions. Overall, these results demonstrate the effectiveness of the implemented approach in this housing price prediction task.")
print("This conclusion is brief, highlights the key metrics (R-squared, MSE, RMSE), and emphasizes the **good** overall performance")
# Conclusion
#
# This project successfully implemented a linear regression model with polynomial features to predict California housing prices. The model demonstrated promising results with a Train R-squared of 0.6966 and a Test R-squared of 0.7006, indicating a good fit to the data. The model's performance was further evaluated using Mean Squared Error (MSE) of 2402131680.745 and Root Mean Squared Error (RMSE) of 49011.546, suggesting relatively accurate predictions. Overall, these results demonstrate the effectiveness of the implemented approach in this housing price prediction task.
#
# This conclusion is brief, highlights the key metrics (R-squared, MSE, RMSE), and emphasizes the "good" overall performance.


