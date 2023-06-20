# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 21:57:22 2023

@author: User
"""


#%% EXPLANATORY VARS.

import pandas as pd

df1 = pd.read_csv("C:\\Users\\User\\Desktop\\UNIVERSITY\\RP\\data_exploration\\profilicId.csv")
df2 = pd.read_excel("C:\\Users\\User\\Desktop\\UNIVERSITY\\RP\\data_exploration\\personal_info.xlsx")
#df3 = pd.read_excel("C:\\Users\\User\\Desktop\\UNIVERSITY\\RP\\data_exploration\\post_questionaire.xlsx", dtype='object')
df4 = pd.read_excel("C:\\Users\\User\\Desktop\\UNIVERSITY\\RP\\data_exploration\\MEMO_participant_group_assignment.xlsx")

merged_df = pd.merge(df1, df2, on='ProfilicID')
merged_df = pd.merge(merged_df, df4, on='ProfilicID')
#merged_df = pd.merge(merged_df, df3, on='ProfilicID')

filtered_df = merged_df[merged_df['ProfilicID'].isin(df1['ProfilicID'])]
filtered_df = filtered_df.dropna(subset=['GENDER', 'Demographic', 'age', 'online_meetings_experience'])
filtered_df['Group'] = filtered_df['group'].copy()
filtered_df = filtered_df.drop_duplicates()



# Keep only the profilicId, group, online_meetings_experience, age,
# Covid-19_affected_group, Covid-19_affected_group_extra, Demographic,
# Perceived_group, GENDER.
final_data = filtered_df[['ProfilicID', 'Group','english_fluency', 'country_of_residence',
'Covid-19_affected_group', 'Demographic', 'Perceived_group', 'online_meetings_experience', 'age', 'GENDER']]
final_data = final_data.drop_duplicates()
#%%


#%% TARGET VAR.

import pandas as pd
df_ratings = pd.read_csv("C:\\Users\\User\\Desktop\\UNIVERSITY\\RP\\data_exploration\\output.csv") # 11614 rows

# overlapping df
df_ratings_duplicates = df_ratings[df_ratings.duplicated(subset=['Start Time (ms)', 'End Time (ms)', 'Group', 'Session'], keep=False)] # has 3247 rows

# Build the dataset, dealing with the overlapped segments
# annotated by different annotators by either keeping that value if it is equal
# or keeping the mean otherwise

# Group the dataframe by the columns that contain duplicate values
df_ratings_overlap = df_ratings_duplicates[['Start Time (ms)', 'End Time (ms)', 'Group', 'Session', 'Involvement']]
df_ratings_overlap = df_ratings_overlap.groupby(['Start Time (ms)', 'End Time (ms)', 'Group', 'Session']).mean() # are 1599 duplicates
# drop all duplicates from the df containing all annotations
final_df_1 = df_ratings.drop_duplicates(subset=['Start Time (ms)', 'End Time (ms)', 'Group', 'Session'], keep=False) # 8367 unique rows
# add to the df with all unique values the overlapping annotations 
final_df = pd.concat([final_df_1, df_ratings_overlap]) # 9966 = 8367 unique + 1599 overlapped


#%% WEIGHTED AVERAGE INVOLVEMENT FOR EACH GROUP (MEAN OF EACH SESSION MEAN)

# Calculate the weighted average involvement
grouped_df = final_df.groupby(['Group', 'Session'])

# Create a new DataFrame with the weighted average involvement
mean_df = pd.DataFrame({'Involvement': grouped_df['Involvement'].mean()}).reset_index()

# Calculate the involvement for each group over all sessions
group_involvement_df = mean_df.groupby('Group')['Involvement'].mean().reset_index()

#%%

#%% LENGTH OF PEAK INVOLVMENT INSTEAD OF MEAN INVOLVEMENT

# # Calculate the count of peak values (e.g., number of 5s) for each session in each group
# grouped_df = final_df[final_df['Involvement'] > 3].groupby(['Group', 'Session'])

# # Create a new DataFrame with the count of peak values
# count_df = pd.DataFrame({'Involvement': grouped_df['Involvement'].count()}).reset_index()

# # Calculate the involvement for each group over all sessions
# group_involvement_df = round(count_df.groupby('Group')['Involvement'].mean()).reset_index()





#%% PREPROCESSED DATASET MERGE


# MERGING EXPLANATORY AND TARGET VARS ON 'GROUP'

df = pd.merge(final_data, group_involvement_df, on='Group') # has only 40 entries as there are 3 people in the explanatory dataset from grp 1
# df['Mean_Involvement'] = df['Mean_Involvement'].round(2)
df['age'] = df['age'].astype(int)

#%% DATA VISUALIZATION

# # =============================================================================
# # VISUALIZE THE DATA SET
# # =============================================================================
# import matplotlib.pyplot as plt
# import seaborn as sns
# categorical = ['Demographic', 'GENDER', 'online_meetings_experience']

# # Plot the involvement
# sns.distplot(df['Mean_Involvement'])
# plt.grid(False)
# plt.show()

# # Plot the age
# sns.distplot(df.age)
# plt.grid(False)
# plt.show()

# # Calculate the weighted average involvement
# grouped_annotator_df = final_df.groupby(['Group', 'Session', 'Annotator'])

# # Create a new DataFrame with the weighted average involvement
# mean_annotator_df = pd.DataFrame({'Mean_Involvement': grouped_annotator_df['Involvement'].mean()}).reset_index()

# # Calculate the involvement for each group over all sessions
# annotator_involvement_df = mean_annotator_df.groupby(['Group', 'Annotator'])['Mean_Involvement'].mean().reset_index()
# sns.catplot(data=annotator_involvement_df, x="Group", y="Mean_Involvement", hue="Annotator", kind="swarm")
# plt.show()

# # Plot the age
# sns.catplot(data=df, x="Group", y="Mean_Involvement", hue="age", kind="swarm")
# plt.show()

# # Plot the gender
# sns.catplot(data=df, x="Group", y="Mean_Involvement", hue="GENDER", kind="swarm")
# plt.show()

# # Plot the demographic
# sns.catplot(data=df, x="Group", y="Mean_Involvement", hue="Demographic", kind="swarm")
# plt.show()

# # Plot the virtual experience
# sns.catplot(data=df, x="Group", y="Mean_Involvement", hue="online_meetings_experience", kind="swarm")
# plt.show()

# # Plot the demographics in each groups
# g = sns.catplot(data=df, x="Group", y="Demographic", kind="swarm")
# g.despine(left=True)
# plt.show()

# # Plot the correalation between the age and the involvement
# dataset = pd.DataFrame()
# dataset['age'] = df.age
# dataset['Mean_Involvement']  = df.Mean_Involvement
# dataset['Involvement'] = df['Mean_Involvement']
# sns.lmplot(x = "age", y = "Mean_Involvement", data=dataset)
# plt.show()

# # Plot the demographics involvement based on gender
# g = sns.catplot(
#     data=df, x="GENDER", y="Mean_Involvement", hue='Group', col="Demographic",
#     kind="bar", col_wrap=3,
# )
# g.set_axis_labels("", "Involvement")
# g.set_xticklabels(["Men", "Women"])
# g.set_titles("{col_name} {col_var}")
# g.despine(left=True)
# plt.show()

# # Plot the demographics involvement based on groups
# g = sns.catplot(
#     data=df, x="Group", y="Mean_Involvement", col="Demographic",
#     kind="bar", col_wrap=3, legend=True
# )
# g.set_axis_labels("Group", "Involvement")
# g.set_titles("{col_name} {col_var}")
# g.despine(left=True)
# plt.show()

#%% CATEGORICAL DATA ENCODING

corr_df = df[['age','GENDER', 'Demographic', 'online_meetings_experience', 'Group', 'Involvement']]

# ONE HOT ENCODER for categorical data
encoded_df = pd.get_dummies(corr_df, drop_first=True)

# Nicely ordered encoded dataframe is encoded.

encoded = pd.DataFrame()
encoded[['age', 'Gender']] = encoded_df[['age', 'GENDER_Male']]
encoded['middle'] = encoded_df['Demographic_middle'].copy()
encoded['older'] = encoded_df['Demographic_older'].copy()
encoded['parent'] = encoded_df['Demographic_parent'].copy()
encoded['student'] = encoded_df['Demographic_student'].copy()
encoded['virtual_experience_Regular'] = encoded_df["online_meetings_experience_I have online meetings on a regular basis"].copy()
encoded['virtual_experience_Previous'] = encoded_df["online_meetings_experience_I've had online meetings before"].copy()
encoded['Group'] = encoded_df["Group"]
encoded['Involvement'] = encoded_df['Involvement'].copy()

# !!! DO NOT SCALE DATA BEFORE SPLITTING IT AS IT MAY LEAK DATA BETWEEN TRAIN AND TEST SETS

dataheat = encoded[['Gender', 'middle', 'older',
        'parent', 'student', 'virtual_experience_Regular',
        "virtual_experience_Previous", 'age', 'Involvement']]

# Heatmap for all initial values
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 8))
sns.heatmap(dataheat.corr(), annot=True, cmap='coolwarm', square=True)
plt.title('Correlation Heatmap with all initial variables.')
plt.show()

# =============================================================================
# CALCULATE VIF
# FIND THRESHOLD FOR VIF TO DROP THE COLS WITH HIGHER VIF THAN THE THRESHOLD
# =============================================================================

from statsmodels.stats.outliers_influence import variance_inflation_factor
import numpy as np

# Select the explanatory variables
explanatory_vars = encoded.columns.tolist()
explanatory_vars.remove('Involvement')
X = encoded[explanatory_vars].astype(float)
# Normalize the age
X['age'] = np.log(X['age'])

dataheat = dataheat.drop(columns=['virtual_experience_Regular'])

# WORKING Create a heatmap using seaborn
plt.figure(figsize=(10, 8))
sns.heatmap(dataheat.corr(), annot=True, cmap='coolwarm', square=True)
plt.title('Correlation Heatmap after regular virtual experience removal.')
plt.show()

# dataheat = dataheat.drop(columns=[, 'virtual_experience_Regular']])
dataheat = dataheat.drop(columns=['older'])

# WORKING Create a heatmap using seaborn
plt.figure(figsize=(10, 8))
sns.heatmap(dataheat.corr(), annot=True, cmap='coolwarm', square=True)
plt.title('Correlation Heatmap after older and regular virtual experience removal.')
plt.show()

# remove the older column as their VIF score is higher than 10
X = X.drop(columns=['older', 'virtual_experience_Regular'])
vif = pd.DataFrame()
vif["Variable"] = X.columns
vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
#%% CROSS VALIDATION FOR DECISION TREE, LR, AND RANDOM FOREST
from math import sqrt
from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import statsmodels.api as sm
import statsmodels.formula.api as smf

from sklearn.metrics import mean_absolute_error, median_absolute_error, mean_absolute_percentage_error, mean_squared_error


y = encoded['Involvement']

formula = 'Involvement ~ 1 + age + Gender + middle + parent + student + virtual_experience_Previous'

# =============================================================================
# CROSS VALIDATION FOR LR. DT. RF USING SKLEARN SCORES
# =============================================================================

# Initialize the models
linear_model = LinearRegression()
tree_model = DecisionTreeRegressor()
forest_model = RandomForestRegressor()

# Define the number of folds for cross-validation
k = 12

# Perform k-fold cross-validation for linear regression
linear_scores = cross_val_score(linear_model, X, y, cv=k, scoring='neg_mean_squared_error')
linear_rmse = (-linear_scores.mean()) ** 0.5

# Perform k-fold cross-validation for decision tree regression
tree_scores = cross_val_score(tree_model, X, y, cv=k, scoring='neg_mean_squared_error')
tree_rmse = (-tree_scores.mean()) ** 0.5

# Perform k-fold cross-validation for random forest regression
forest_scores = cross_val_score(forest_model, X, y, cv=k, scoring='neg_mean_squared_error')
forest_rmse = (-forest_scores.mean()) ** 0.5

# Perform cross-validation and generate predicted values for each fold
linear_pred = cross_val_predict(linear_model, X, y, cv=k)
tree_pred = cross_val_predict(tree_model, X, y, cv=k)
forest_pred = cross_val_predict(forest_model, X, y, cv=k)

# Calculate the residuals (errors) between the predicted values and the actual values
linear_residuals = y - linear_pred
tree_residuals = y - tree_pred
forest_residuals = y - forest_pred

# Calculate MAPE
linear_mape = np.mean(np.abs(linear_residuals / y)) * 100
tree_mape = np.mean(np.abs(tree_residuals / y)) * 100
forest_mape = np.mean(np.abs(forest_residuals / y)) * 100

# Calculate MAE
linear_mae = mean_absolute_error(y, linear_pred)
tree_mae = mean_absolute_error(y, tree_pred)
forest_mae = mean_absolute_error(y, forest_pred)

# Calculate MedAE
linear_medae = median_absolute_error(y, linear_pred)
tree_medae = median_absolute_error(y, tree_pred)
forest_medae = median_absolute_error(y, forest_pred)

print("Linear Regression RMSE:", linear_rmse)
print("Decision Tree Regression RMSE:", tree_rmse)
print("Random Forest Regression RMSE:", forest_rmse)

# =============================================================================
# CROSS VALIDATION FOR GLMM, LR, DT, RF ADJUSTED CODE
# =============================================================================

# Create empty lists to store the predicted values and residuals
# glmm_pred = []
# glmm_residuals = []
glmm_mae_scores = []
glmm_mape_scores = []
glmm_medae_scores = []
glmm_rmse_scores = []

lr_mae_scores = []
lr_mape_scores = []
lr_medae_scores = []
lr_rmse_scores = []

dt_mae_scores = []
dt_mape_scores = []
dt_medae_scores = []
dt_rmse_scores = []

rf_mae_scores = []
rf_mape_scores = []
rf_medae_scores = []
rf_rmse_scores = []

# Perform cross-validation
kf = KFold(n_splits=k, shuffle=True, random_state=47)
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    train_data_cross = pd.DataFrame()
    train_data_cross = X_train.copy()
    train_data_cross['Involvement'] = y_train

    # Fit the MixedLM model on the training data
    glmm_model = sm.MixedLM(y_train, X_train, groups=X_train['Group']).fit()
    # glmm_model = smf.mixedlm(formula=formula, data=train_data_cross, groups=train_data_cross['Group']).fit()
    print(glmm_model.summary())
    
    # Predict the values for the testing data
    y_pred = glmm_model.predict(X_test)
    # Calculate evaluation metrics for the current fold
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    medae = median_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # Append the scores to the lists
    glmm_mae_scores.append(mae)
    glmm_mape_scores.append(mape)
    glmm_medae_scores.append(medae)
    glmm_rmse_scores.append(rmse)

    # # Append the predicted values and residuals to the respective lists
    # glmm_pred.extend(y_pred)
    # glmm_residuals.extend(y_test - y_pred)
    
    residuals = y_test - y_pred
    # Plot residuals against predicted values
    plt.scatter(y_pred, residuals)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs. Predicted Values for GLMM')
    plt.show()
    
    
    # Fit the Linear Regression model on the training data
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)

    # Fit the Decision Tree model on the training data
    dt_model = DecisionTreeRegressor()
    dt_model.fit(X_train, y_train)

    # Fit the Random Forest model on the training data
    rf_model = RandomForestRegressor()
    rf_model.fit(X_train, y_train)

    # Make predictions on the test data for each model
    lr_pred = lr_model.predict(X_test)
    dt_pred = dt_model.predict(X_test)
    rf_pred = rf_model.predict(X_test)
    
    residuals_lr = y_test - lr_pred
    residuals_dt = y_test - dt_pred
    residuals_rf = y_test - rf_pred
    # Plot residuals for LR
    plt.scatter(lr_pred, residuals_lr)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs. Predicted Values for LR')
    plt.show()
    # Plot residuals for DT
    plt.scatter(dt_pred, residuals_dt)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs. Predicted Values for DT')
    plt.show()
    # Plot residuals for RF
    plt.scatter(rf_pred, residuals_rf)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs. Predicted Values for RF')
    plt.show()

    # Calculate evaluation metrics for each model
    lr_mae = mean_absolute_error(y_test, lr_pred)
    lr_mape = mean_absolute_percentage_error(y_test, lr_pred)
    lr_medae = median_absolute_error(y_test, lr_pred)
    lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))

    dt_mae = mean_absolute_error(y_test, dt_pred)
    dt_mape = mean_absolute_percentage_error(y_test, dt_pred)
    dt_medae = median_absolute_error(y_test, dt_pred)
    dt_rmse = np.sqrt(mean_squared_error(y_test, dt_pred))

    rf_mae = mean_absolute_error(y_test, rf_pred)
    rf_mape = mean_absolute_percentage_error(y_test, rf_pred)
    rf_medae = median_absolute_error(y_test, rf_pred)
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
    
    # Append the scores to the lists
    lr_mae_scores.append(lr_mae)
    lr_mape_scores.append(lr_mape)
    lr_medae_scores.append(lr_medae)
    lr_rmse_scores.append(lr_rmse)
    
    dt_mae_scores.append(dt_mae)
    dt_mape_scores.append(dt_mape)
    dt_medae_scores.append(dt_medae)
    dt_rmse_scores.append(dt_rmse)
    
    rf_mae_scores.append(rf_mae)
    rf_mape_scores.append(rf_mape)
    rf_medae_scores.append(rf_medae)
    rf_rmse_scores.append(rf_rmse)


# Print the RMSE for each model
performancedf = pd.DataFrame(columns=['Model', 'RMSE', 'MAPE', 'MAE', 'MedAE'])

# Calculate the mean and standard deviation of evaluation metric scores
mae_mean = np.mean(glmm_mae_scores)
mae_std = np.std(glmm_mae_scores)
mape_mean = np.mean(glmm_mape_scores)
mape_std = np.std(glmm_mape_scores)
medae_mean = np.mean(glmm_medae_scores)
medae_std = np.std(glmm_medae_scores)  
rmse_mean = np.mean(glmm_rmse_scores)
rmse_std = np.std(glmm_rmse_scores)   
performancedf.loc[0] = ["GLMM", rmse_mean, mape_mean, mae_mean, medae_mean]
 
mae_mean = np.mean(lr_mae_scores)
mae_std = np.std(lr_mae_scores)
mape_mean = np.mean(lr_mape_scores)
mape_std = np.std(lr_mape_scores)
medae_mean = np.mean(lr_medae_scores)
medae_std = np.std(lr_medae_scores)  
rmse_mean = np.mean(lr_rmse_scores)
rmse_std = np.std(lr_rmse_scores)  
performancedf.loc[1] = ["Linear Regression", rmse_mean, mape_mean, mae_mean, medae_mean]

mae_mean = np.mean(dt_mae_scores)
mae_std = np.std(dt_mae_scores)
mape_mean = np.mean(dt_mape_scores)
mape_std = np.std(dt_mape_scores)
medae_mean = np.mean(dt_medae_scores)
medae_std = np.std(dt_medae_scores)  
rmse_mean = np.mean(dt_rmse_scores)
rmse_std = np.std(dt_rmse_scores)  
performancedf.loc[2] = ["Decision Tree", rmse_mean, mape_mean, mae_mean, medae_mean]

mae_mean = np.mean(rf_mae_scores)
mae_std = np.std(rf_mae_scores)
mape_mean = np.mean(rf_mape_scores)
mape_std = np.std(rf_mape_scores)
medae_mean = np.mean(rf_medae_scores)
medae_std = np.std(rf_medae_scores)  
rmse_mean = np.mean(rf_rmse_scores)
rmse_std = np.std(rf_rmse_scores)  
performancedf.loc[3] = ["Random Forest", rmse_mean, mape_mean, mae_mean, medae_mean]

#%% TRAIN TEST SPLIT
import numpy as np
from sklearn.model_selection import train_test_split

# Cannot split the data set 70:30 as the GLMM needs to have 
# the same # of unique groups in both training and testing.
# For some reason keeping the same number in both with stratify does not 
# satisfy the GLMM model.
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=47)


# =============================================================================
# splitting such as we have the same number of groups in train as in test set
# =============================================================================

data= pd.DataFrame()
data = X.copy()
data['Involvement'] = y

# ValueError: The least populated class in y has only 1 member, which is too few. The minimum number of groups for any class cannot be less than 2.
data = data[data['Group'] != 13]
y = data['Involvement']
X = data.drop(columns=['Involvement'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, stratify=X['Group'], random_state=47)

train_data = pd.DataFrame()
train_data = X_train.copy()
train_data['Involvement'] = y_train

#%%


#%% PREDICTION MODELS

# from sklearn.metrics import r2_score 

# NORMALIZE DATA AFTER SPLIT when using age in explanatory data set
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

normalized_x_train = pd.DataFrame(
    scaler.fit_transform(X_train),
    columns = X_train.columns
)
normalized_x_test = pd.DataFrame(
    scaler.transform(X_test),
    columns = X_test.columns
)

#%% DECISION TREE REGRESSION MODEL

from sklearn import tree

clf_regressor_model = tree.DecisionTreeRegressor()
clf_regressor_model = clf_regressor_model.fit(normalized_x_train, y_train)
clf_predictions = clf_regressor_model.predict(normalized_x_test)

# Reset indexes
X_test.reset_index(drop=True, inplace=True)
y_test.reset_index(drop=True, inplace=True)

# Calculate residuals
clf_residuals = y_test - clf_predictions
# Plot residuals against predicted values
plt.scatter(clf_predictions, clf_residuals)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs. Predicted Values for Decision Tree Regression')
plt.show()


clf_errors = abs(clf_predictions - y_test)
# Store the RMSE
RMSE = np.sqrt(mean_squared_error(y_test,clf_predictions))
# Calculate mean absolute percentage error (MAPE)
# mape = 100 * (clf_errors / y_test)
mape = mean_absolute_percentage_error(y_test, clf_predictions)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
# Calculate MAE
mae = mean_absolute_error(y_test, clf_predictions)
# Calculate MedAE
medae = median_absolute_error(y_test, clf_predictions)
# Adjusted R-squared
# n = clf_predictions.shape[0]
# k = X_train.shape[1]
# r2 = r2_score(y_test,clf_predictions)
# adj_r_sq = 1 - (1 - r2)*(n-1)/(n-1-k)

results = pd.DataFrame()
results["Method"] = ["Decision Tree"]
results["RMSE"] = RMSE
results['MAPE'] = mape
results['MAE'] = mae
results['MeDAE'] = medae

#%% RANDOM FOREST REGRESSOR MODEL

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
rf.fit(normalized_x_train, y_train);
rf_predictions = rf.predict(normalized_x_test)

# Calculate residuals
rf_residuals = y_test - rf_predictions
# Plot residuals against predicted values
plt.scatter(rf_predictions, rf_residuals)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs. Predicted Values for Random Forest Regression')
plt.show()


# Calculate the absolute errors
rf_errors = abs(rf_predictions - y_test)
# RMSE
rmse = sqrt(mean_squared_error(y_test,rf_predictions))
# Calculate mean absolute percentage error (MAPE)
# mape = 100 * (rf_errors / y_test)
mape = mean_absolute_percentage_error(y_test, rf_predictions)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
# Calculate MAE
mae = mean_absolute_error(y_test, rf_predictions)
# Calculate MedAE
medae = median_absolute_error(y_test, rf_predictions)
results.loc[1] = ["Random Forest", rmse, mape, mae, medae]

#%% LINEAR REGRESSION NEW

# import statsmodels.formula.api as smf
# formula = 'Involvement ~ age + Gender + middle + older + parent + student + virtual_experience_Previous'

mod = smf.ols(formula=formula, data=train_data).fit()
print('Regression summary')
print(mod.summary())

# PREDICT on testing set
lr_predictions = mod.predict(normalized_x_test)

# Calculate residuals
lr_residuals = y_test - lr_predictions
# Plot residuals against predicted values
plt.scatter(lr_predictions, lr_residuals)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs. Predicted Values for Linear Regression')
plt.show()

lr_errors = abs(lr_predictions - y_test)
# Store the RMSE
RMSE = sqrt(mean_squared_error(y_test, lr_predictions))
# Calculate mean absolute percentage error (MAPE)
# mape = 100 * (lr_errors / y_test)
mape = mean_absolute_percentage_error(y_test, lr_predictions)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
# Calculate MAE
mae = mean_absolute_error(y_test, lr_predictions)
# Calculate MedAE
medae = median_absolute_error(y_test, lr_predictions)
results.loc[2] = ['Linear Regression', RMSE, mape, mae, medae]

#%% Predictions plots


# # PREDICTION ERROR PLOT
# from sklearn.linear_model import Lasso
# from yellowbrick.regressor import PredictionError

# # Instantiate the linear model and visualizer
# model = Lasso()
# visualizer = PredictionError(model)

# visualizer.fit(normalized_x_train, y_train)  # Fit the training data to the model
# visualizer.score(normalized_x_test, y_test)  # Evaluate the model on the test data
# visualizer.poof()                    # Draw/show/poof the data

# ax = sns.residplot(x = "age_scaled", y= "Mean_Involvement", data = encoded, lowess = True)
# ax.set(ylabel='Observed - Prediction')
# plt.show()

# # RESIDUALS GRAPH
# from sklearn.linear_model import Ridge
# from yellowbrick.regressor import ResidualsPlot

# # Instantiate the linear model and visualizer
# model = Ridge()
# visualizer = ResidualsPlot(model)

# visualizer.fit(normalized_x_train, y_train)  # Fit the training data to the model
# visualizer.score(normalized_x_test, y_test)  # Evaluate the model on the test data
# visualizer.poof()                    # Draw/show/poof the data

#%% GENERALISED MIXED MODEL REGRESSION

# import researchpy as rp
# rp.codebook(train_data)
# data = pd.DataFrame()
# data = X.copy()
# data['Involvement'] = y.copy()
# rp.codebook(data)

# import statsmodels.formula.api as smf
# formula = 'Involvement ~ 1 + age + Gender + middle + parent + student + virtual_experience_Previous'

mod = smf.mixedlm(formula=formula, data=train_data, groups=train_data['Group']).fit()
print('GLMM summary')
print(mod.summary())

# Predict on the test data
mod_predictions = mod.predict(X_test)

# Calculate residuals
mod_residuals = y_test - mod_predictions
# Plot residuals against predicted values
plt.scatter(mod_predictions, mod_residuals)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs. Predicted Values for GLMM')
plt.show()


mod_errors = abs(mod_predictions - y_test)
# RMSE
rmse = sqrt(mean_squared_error(y_test,mod_predictions))
# Calculate mean absolute percentage error (MAPE)
# mape = 100 * (mod_errors / y_test)
mape = mean_absolute_percentage_error(y_test, mod_predictions)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
# Calculate MAE
mae = mean_absolute_error(y_test, mod_predictions)
# Calculate MedAE
medae = median_absolute_error(y_test, mod_predictions)

results.loc[3] = ["GLMM", rmse, mape, mae, medae]

#%% LMM same results as GLMM

# from statsmodels.regression.mixed_linear_model import MixedLM
# import statsmodels.api as sm

# group = X_train['Group']
# X_train_fixed = pd.DataFrame()
# X_train_fixed = X_train[['age', 'Gender', 'middle', 'parent', 'student', 'virtual_experience_Previous']]
# X_train_fixed = sm.add_constant(X_train_fixed)

# model = MixedLM(y_train, X_train_fixed, groups=group).fit()

# # Get the estimated fixed effects coefficients
# fixed_effects = model.params
# print('Fixed effects for MixedLm (model): ') 
# print(fixed_effects)

# print('LMM summary (model):')
# print(model.summary())

# # Predict on the test data
# X_test = X_test.drop(columns=['Group'])
# X_test = sm.add_constant(X_test)
# lmm_predictions = model.predict(X_test)

# # Calculate residuals
# lmm_residuals = y_test - lmm_predictions
# # Plot residuals against predicted values
# plt.scatter(lmm_predictions, lmm_residuals)
# plt.axhline(y=0, color='red', linestyle='--')
# plt.xlabel('Predicted Values')
# plt.ylabel('Residuals')
# plt.title('Residuals vs. Predicted Values for LMM model')
# plt.show()


# lmm_errors = abs(lmm_predictions - y_test)
# # RMSE
# rmse = sqrt(mean_squared_error(y_test,lmm_predictions))
# # Calculate mean absolute percentage error (MAPE)
# # mape = 100 * (lmm_errors / y_test)
# mape = mean_absolute_percentage_error(y_test, lmm_predictions)
# # Calculate and display accuracy
# accuracy = 100 - np.mean(mape)
# # Calculate MAE
# mae = mean_absolute_error(y_test, lmm_predictions)
# # Calculate MedAE
# medae = median_absolute_error(y_test, lmm_predictions)
# results.loc[5] = ["LMM model", rmse, mape, mae, medae]

#%% LMM RESULTS BASED ON THE ENTIRE DATASET

# group = X['Group']
# X_fixed = pd.DataFrame()
# X_fixed = X[['age', 'Gender', 'middle', 'parent', 'student', 'virtual_experience_Previous', 'Group']]
# X_fixed = pd.DataFrame(
#     scaler.fit_transform(X_fixed),
#     columns = X_fixed.columns
# )
# X_fixed = sm.add_constant(X_fixed)
# y = y.reset_index(drop=True)

# model = MixedLM(y, X_fixed, groups=group).fit()

# Get the estimated fixed effects coefficients
# fixed_effects = model.params
# print('Fixed effects for MixedLm (model) on the whole data set: ') 
# print(fixed_effects)

# print('LMM summary (model) on the whole data set:')
# print(model.summary())

#%% GLMM on the categorical variables before encoding them

# formula = 'Involvement ~ age + C(GENDER) + C(Demographic) + C(online_meetings_experience)'

# mod1 = smf.mixedlm(formula=formula, data=corr_df, groups='Group').fit()
# print('MixedLm (Group) summary')
# print(mod1.summary())
#%%
#%% LINEAR REGRESSION OLD

# import matplotlib.pyplot as plt
# from sklearn.linear_model import LinearRegression

# # construct our linear regression model and fit training data
# lr = LinearRegression(fit_intercept=True).fit(normalized_x_train, y_train)

# # # and let's plot what this relationship looks like between variables and involvement
# # xfit = np.linspace(-3, 3, 1000)
# # yfit = lr.predict(xfit[:, np.newaxis])
# # plt.scatter(X_train, y)
# # plt.plot(xfit, yfit);
# # plt.xlabel("Explanatory variables")
# # plt.ylabel("Involvement")

# # PREDICT on testing set
# lr_predictions = lr.predict(normalized_x_test)

# # Store the slope and interceipt
# lo_slope = lr.coef_[0]
# lr_interceipt = lr.intercept_

# # Calculate residuals
# lr_residuals = y_test - lr_predictions
# # Plot residuals against predicted values
# plt.scatter(lr_predictions, lr_residuals)
# plt.axhline(y=0, color='red', linestyle='--')
# plt.xlabel('Predicted Values')
# plt.ylabel('Residuals')
# plt.title('Residuals vs. Predicted Values for Linear Regression')
# plt.show()

# lr_errors = abs(lr_predictions - y_test)

# # Store the RMSE
# RMSE = sqrt(mean_squared_error(y_test, lr_predictions))
# r2 = r2_score(y_test,lr_predictions)
# # Calculate mean absolute percentage error (MAPE)
# mape = 100 * (lr_errors / y_test)
# # Calculate and display accuracy
# accuracy = 100 - np.mean(mape)

# results.loc[2] = ["Linear Regression", RMSE, r2, accuracy]
#%% OLD GLMM

# import statsmodels.api as sm

# # Define the formula for the GLMM
# formula = 'Mean_Involvement ~ Gender + middle + older + parent + student + virtual_experience_Regular + virtual_experience_Previous + C(Group)'
# # formula = "Mean_Involvement ~ Female + Male + business + middle + older + parent + student + virtual_experience_Never + virtual_experience_Regular + virtual_experience_Previous + C(Group)"


# # Reset indexes
# # X_train.reset_index(drop=True, inplace=True)
# # y_train.reset_index(drop=True, inplace=True)

# # Create a DataFrame with both predictors and target variable
# train_data = X_train.copy()
# train_data['Group'] = train_data['Group'].astype('category')
# train_data['Mean_Involvement'] = y_train

# # Fit the GLMM model
# # glmm = smf.glm(formula=formula, data=train_data, family=sm.families.Gaussian()).fit()
# # glmm = smf.mixedlm(formula, train_data, groups=train_data['Group']).fit()
# glmm = sm.GLM(y_train, X_train, family=sm.families.Gaussian()).fit()

# print(glmm.summary())

# # Predict on the test data
# glmm_predictions = glmm.predict(X_test)

# # Calculate residuals
# glmm_residuals = y_test - glmm_predictions
# # Plot residuals against predicted values
# plt.scatter(glmm_predictions, glmm_residuals)
# plt.axhline(y=0, color='red', linestyle='--')
# plt.xlabel('Predicted Values')
# plt.ylabel('Residuals')
# plt.title('Residuals vs. Predicted Values for GLMM')
# plt.show()


# glmm_errors = abs(glmm_predictions - y_test)

# # RMSE
# rmse = sqrt(mean_squared_error(y_test,glmm_predictions))
# # R2
# r2 = r2_score(y_test,glmm_predictions)
# # Calculate mean absolute percentage error (MAPE)
# mape = 100 * (glmm_errors / y_test)
# # Calculate and display accuracy
# accuracy = 100 - np.mean(mape)

# results.loc[5] = ["GLMM", rmse, r2, accuracy]

#%% Gaussian Process Regression

# from sklearn.gaussian_process import GaussianProcessRegressor
# from sklearn.gaussian_process.kernels import RBF

# # Create the Gaussian process regression model
# kernel = RBF(length_scale=1.0)
# gp = GaussianProcessRegressor(kernel=kernel)

# # Fit the model to the training data
# gp.fit(X_train, y_train)

# # Predict using the trained model
# gp_predictions, y_pred_std = gp.predict(X_test, return_std=True)

# # Calculate residuals
# gp_residuals = y_test - gp_predictions
# # Plot residuals against predicted values
# plt.scatter(gp_predictions, gp_residuals)
# plt.axhline(y=0, color='red', linestyle='--')
# plt.xlabel('Predicted Values')
# plt.ylabel('Residuals')
# plt.title('Residuals vs. Predicted Values for Gaussian Process Regression')
# plt.show()


# gp_errors = abs(gp_predictions - y_test)

# # RMSE
# rmse = sqrt(mean_squared_error(y_test,gp_predictions))
# # R2
# r2 = r2_score(y_test,gp_predictions)
# # Calculate mean absolute percentage error (MAPE)
# mape = 100 * (gp_errors / y_test)
# # Calculate and display accuracy
# accuracy = 100 - np.mean(mape)

# results.loc[5] = ["Gaussian Process Regression", rmse, r2, accuracy]

#%% GPBoost

# import gpboost as gpb

# # Specify the parameters for the GPBoost model
# params = {
#     'objective': 'regression',
#     'task': 'train',
#     'learning_rate': 0.1,
#     'num_iterations': 100,
#     'num_leaves': 31,
#     'verbose': 0
# }

# # Create the GPBoost dataset
# group_sizes = X_train.groupby('Group').size().values
# train_dataset = gpb.Dataset(X_train, y_train, group=group_sizes)


# # Train the GPBoost model
# model = gpb.train(params, train_dataset)

# # Make predictions on the test data
# gpb_predictions = model.predict(X_test)



# # Calculate residuals
# gpb_residuals = y_test - gpb_predictions
# # Plot residuals against predicted values
# plt.scatter(gpb_predictions, gpb_residuals)
# plt.axhline(y=0, color='red', linestyle='--')
# plt.xlabel('Predicted Values')
# plt.ylabel('Residuals')
# plt.title('Residuals vs. Predicted Values for GPBoost')
# plt.show()


# gpb_errors = abs(gpb_predictions - y_test)

# # RMSE
# rmse = sqrt(mean_squared_error(y_test,gpb_predictions))
# # R2
# r2 = r2_score(y_test,gpb_predictions)
# # Calculate mean absolute percentage error (MAPE)
# mape = 100 * (gpb_errors / y_test)
# # Calculate and display accuracy
# accuracy = 100 - np.mean(mape)

# results.loc[6] = ["BPBoost", rmse, r2, accuracy]
#%%


#%% GLMM first attempt

# import researchpy as rp
# import statsmodels.formula.api as smf

# encoded['Mean_Involvement'] = y
# rp.codebook(X)
# rp.summary_cont(encoded.groupby(["age", "Gender_Male"])["Mean_Involvement"])

# data =pd.DataFrame()
# data = normalized_x_train.copy()
# y_train = y_train.reset_index()
# data['Mean_Involvement'] = y_train['Mean_Involvement']

# # fit our model
# md = smf.mixedlm("Mean_Involvement ~ Gender_Male", data, groups=data["Group"])
# mdf = md.fit()
# print(mdf.summary())

# # Plot the predictions
# performance = pd.DataFrame()
# performance["residuals"] = mdf.resid.values
# performance["gender"] = data.Gender_Male
# performance["predicted"] = mdf.predict(X_test)

# sns.lmplot(x = "predicted", y = "residuals", data = performance)

# ax = sns.residplot(x = "gender", y = "residuals", data = performance, lowess=True)
# ax.set(ylabel='Observed - Prediction')
# plt.show()

# mdf_predict = mdf.predict(X_test)
# # Calculate the absolute errors
# errors = abs(mdf_predict - y_test)

# # Store the RMSE
# RMSE = sqrt(mean_squared_error(y_test, mdf_predict))
# # R2
# r2 = r2_score(y_test,mdf_predict)
# # Calculate mean absolute percentage error (MAPE)
# mape = 100 * (errors / y_test)
# # Calculate and display accuracy
# accuracy = 100 - np.mean(mape)

# results.loc[5] = ["Mixed_Random_Model", RMSE, r2, accuracy]
#%%

