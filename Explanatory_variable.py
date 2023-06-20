# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

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


# Keep only the profilicId, group, online_meetings_experience, age,
# Covid-19_affected_group, Covid-19_affected_group_extra, Demographic,
# Perceived_group, GENDER.
final_data = filtered_df[['ProfilicID', 'Group', 'online_meetings_experience', 'age','Demographic', 'GENDER']]
final_data = final_data.drop_duplicates()

# =============================================================================
# PLOTS
# 
# # =============================================================================
# # CATEGORICAL DATA PLOTS
# # =============================================================================
# import seaborn as sns
# sns.set(style='whitegrid', palette="deep", font_scale=1.1, rc={"figure.figsize": [8, 5]})
# # plot the gender and demographics (categorical data)
# data = filtered_df.loc[filtered_df['group'] == 15]
# # Convert 'gender' column to categorical data type
# gender= data['GENDER']
# gender = pd.DataFrame(gender)
# 
# # Plot the gender counts using Seaborn
# sns.countplot(x='GENDER', data=gender).set(xlabel='gender in Group 15', ylabel='Count');
# 
# # =============================================================================
# # NUMERICAL DATA PLOTS
# # =============================================================================
# # Keep only the rows where the 'group' column has the value 5
# group_age = filtered_df.loc[filtered_df['group'] == 15]
# # plot the age histogram (numerical variables)
# sns.distplot(
#     group_age['age'], norm_hist=False, kde=False, bins=20, hist_kws={"alpha": 1}
# ).set(xlabel='Age in Group 15', ylabel='Count');
# 
# =============================================================================

# =============================================================================
# ONE HOT ENCODER
# =============================================================================
# data = final_data
# X_enc = data.copy()

# #ONEHOT ENCODING 
# X_enc = pd.get_dummies(X_enc, columns = ['GENDER', 'Demographic', 'online_meetings_experience'], drop_first=True)

# data = data.drop(['ProfilicID','Group', 'age', 'GENDER', 'Demographic', 'online_meetings_experience'], axis=1)

# final_data = pd.concat([data, X_enc], axis=1)



# =============================================================================
# LABEL ENCODER
# from sklearn.preprocessing import LabelEncoder
# label = LabelEncoder()
# 
# label.fit(filtered_df.GENDER)
# filtered_df.GENDER = label.transform(filtered_df.GENDER)
# print(filtered_df.GENDER.head)
# 
# label.fit(filtered_df.Demographic)
# filtered_df.Demographic = label.transform(filtered_df.Demographic)
# print(filtered_df.Demographic.head)
# 
# label.fit(filtered_df.online_meetings_experience)
# filtered_df.online_meetings_experience = label.transform(filtered_df.online_meetings_experience)
# =============================================================================


# =============================================================================
# TARGET VARIABLE EXPLORATION
# =============================================================================

df_ratings = pd.read_csv("C:\\Users\\User\\Desktop\\UNIVERSITY\\RP\\data_exploration\\output.csv")

# overlapping df
df_ratings = df_ratings[df_ratings.duplicated(subset=['Start Time (ms)', 'End Time (ms)', 'Group', 'Session'], keep=False)]
df_ratings = df_ratings[['Start Time (ms)', 'End Time (ms)', 'Group', 'Session', 'Annotator', 'Involvement']]
#columns needed: start time, end time, annotator, rating

# =============================================================================
# ICC INTER-ANNOTATOR AGREEMENT
# =============================================================================

df_ratings['Group_session'] = "g" + df_ratings['Group'].astype(str) +"-s"+ df_ratings['Session'].astype(str)
data = df_ratings[['Group_session', 'Annotator', 'Involvement']]
data12 = data.loc[data['Annotator'].isin([1, 2])]
data12 = data12.loc[data12.duplicated()]
data13 = data.loc[data['Annotator'].isin([1, 3])]
data13 = data13.loc[data13.duplicated()]
data14 = data.loc[data['Annotator'].isin([1, 4])]
data14 = data14.loc[data14.duplicated()]
data23 = data.loc[data['Annotator'].isin([2, 3])]
data23 = data23.loc[data23.duplicated()]
data24 = data.loc[data['Annotator'].isin([2, 4])]
data24 = data24.loc[data24.duplicated()]
data34 = data.loc[data['Annotator'].isin([3, 4])]
data34 = data34.loc[data34.duplicated()]

import pingouin as pg

# Calculate ICC
icc_result12 = pg.intraclass_corr(data=data12, targets='Group_session', raters='Annotator', ratings='Involvement')
icc_result13 = pg.intraclass_corr(data=data13, targets='Group_session', raters='Annotator', ratings='Involvement')
icc_result14 = pg.intraclass_corr(data=data14, targets='Group_session', raters='Annotator', ratings='Involvement')
icc_result23 = pg.intraclass_corr(data=data23, targets='Group_session', raters='Annotator', ratings='Involvement')
icc_result24 = pg.intraclass_corr(data=data24, targets='Group_session', raters='Annotator', ratings='Involvement')
icc_result34 = pg.intraclass_corr(data=data34, targets='Group_session', raters='Annotator', ratings='Involvement')

# =============================================================================
# Build the dataset, dealing with the overlapped segments
# annotated by different annotators by either keeping that value if it is equal
# or keeping the weighted mean (weights = annotators) otherwise
# =============================================================================

# Group the dataframe by the columns that contain duplicate values
df_ratings = df_ratings[['Start Time (ms)', 'End Time (ms)', 'Group', 'Session', 'Annotator', 'Involvement']] # has 3247 rows
groups = df_ratings.groupby(['Start Time (ms)', 'End Time (ms)', 'Group', 'Session']) # are 1599 duplicates
# duplicates = []

# Iterate over each group
for (start, end, group, session), annotations in groups:
    if len(annotations) > 1:  # Check if there are duplicate rows
        # duplicates.append(annotations)
        
        if not annotations.empty:
            # Check if the duplicated rows are not equal
            if not annotations.equals(annotations.iloc[0]):
                # Calculate the weighted mean
                weighted_mean = (annotations['Involvement'] * annotations['Annotator']).sum() / annotations['Annotator'].sum()
                
                # Update all duplicated rows with the weighted mean
                df_ratings.loc[annotations.index, 'Involvement'] = round(weighted_mean)
            

final_df = df_ratings.drop_duplicates(subset=df_ratings.columns.difference(['Annotator'])) # has 1599 rows, ????so what happens to the extra 49 rows



# =============================================================================
# WEIGHTED AVERAGE INVOLVEMENT FOR EACH SESSION EACH GROUP
# =============================================================================

# Calculate the weighted average involvement
grouped_df = final_df.groupby(['Group', 'Session'])

weighted_average_involvement = grouped_df['Involvement'].sum() / grouped_df['Involvement'].count()

# Create a new DataFrame with the weighted average involvement
weighted_average_df = pd.DataFrame({'Weighted_Involvement': weighted_average_involvement})

# Reset the index of the new DataFrame
weighted_average_df = weighted_average_df.reset_index()

# Calculate the involvement for each group over all sessions
group_involvement_df = weighted_average_df.groupby('Group')['Weighted_Involvement'].mean().reset_index()
group_involvement_df.columns = ['Group', 'Mean_Involvement']
# =============================================================================
# 
# # MEAN FOR EACH SESSION EACH GROUP OF INVOLVEMENT
# # Group the data by 'Group' and 'Session' and calculate the average involvement
# average_involvement = final_df.groupby(['Group', 'Session'])['Involvement'].mean()
# 
# # Reset the index to convert the result to a DataFrame
# average_involvement = average_involvement.reset_index()
# 
# # Rename the column to 'Average_Involvement'
# average_involvement = average_involvement.rename(columns={'Involvement': 'Average_Involvement'})
# =============================================================================





# =============================================================================
# MERGING EXPLANATORY AND TARGET VARS ON 'GROUP'
# =============================================================================

df = pd.merge(final_data, group_involvement_df, on='Group') # ???Why does it not take all rows from final_data
df = df.drop_duplicates()

# =============================================================================
# import seaborn as sns
# # involvement distribution base on age groups
# age = df[['ProfilicID', 'Group', 'Session', 'age', 'Involvement']]
# age = age.drop_duplicates()
# sns.set(style='whitegrid', palette="deep", font_scale=1.1, rc={"figure.figsize": [8, 5]})
# sns.scatterplot(x=age['age'], y=age['Involvement']);
# 
# time =df.loc[df['Group'] == 5]
# time = time.loc[time['Session'] == 1]
# sns.scatterplot(x=time['Start Time (ms)'], y=time['Involvement']);
# g = sns.FacetGrid(time, hue="Annotator")
# g.map(sns.scatterplot, "Start Time (ms)", "Involvement")
# g.add_legend()
# 
# # the distribution of involvement based on groups
# sortedlist = df.groupby(['Group'])['Involvement'].median().sort_values()
# sns.boxplot(x=df['Group'], y=df['Involvement'], order=list(sortedlist.index))
# 
# # conditional plotting each group
# cond_plot = sns.FacetGrid(data=df, col='Group', hue='Annotator', col_wrap=4)
# cond_plot.map(sns.scatterplot, 'Start Time (ms)', 'Involvement');
# cond_plot.add_legend()
# =============================================================================


# Create the df with the columns: 'age', 'GENDER_Female', 'GENDER_Male',
       # 'Demographic_business', 'Demographic_middle', 'Demographic_older',
       # 'Demographic_parent', 'Demographic_student',
       # 'online_meetings_experience_I have never had online meetings',
       # 'online_meetings_experience_I have online meetings on a regular basis',
       # 'online_meetings_experience_I've had online meetings before',
# corr_df = df[['age', 'GENDER_Female', 'GENDER_Male's,
#         'Demographic_business', 'Demographic_middle', 'Demographic_older',
#         'Demographic_parent', 'Demographic_student']]
# corr_df['virtual_experience_Never'] = df[['online_meetings_experience_I have never had online meetings']]
# corr_df['virtual_experience_Previous'] = df[["online_meetings_experience_I've had online meetings before"]]
# corr_df['virtual_experience_Regular'] = df[['online_meetings_experience_I have online meetings on a regular basis']]
# corr_df['involvement'] = df[['Involvement']]

# corr_df['age'] = corr_df['age'].astype(int)

corr_df = df[['age','GENDER', 'Demographic', 'online_meetings_experience', 'Mean_Involvement']]
corr_df['age'] = corr_df['age'].astype(int)

encoded_df = pd.get_dummies(corr_df, drop_first=True)

# =============================================================================
# # the people who've never had virtual meetings before
# f = encoded_df[(encoded_df['online_meetings_experience_I have online meetings on a regular basis'] == False) & (encoded_df["online_meetings_experience_I've had online meetings before"] == False)]
# 
# =============================================================================
# WORKING!
# Calculate the correlation matrix
corr_matrix = encoded_df.corr()
encoded_df = encoded_df.dropna()
encoded_df = encoded_df._get_numeric_data()


# =============================================================================
# # NOT WORKING
# value = 0
# for i in range(len(corr_matrix)):
#     for j in range(i, len(corr_matrix)):
#         value += corr_matrix[i, j]
#         
#         
# 
# # WORKING Create a heatmap using seaborn
# import matplotlib.pyplot as plt
# plt.figure(figsize=(10, 8))
# sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', square=True)
# plt.title('Correlation Heatmap')
# plt.show()
# =============================================================================




# =============================================================================
# CALCULATE VIF
# FIND THRESHOLD FOR VIF TO DROP THE COLS WITH HIGHER VIF THAN THE THRESHOLD
# =============================================================================

from statsmodels.stats.outliers_influence import variance_inflation_factor

# Select the explanatory variables
explanatory_vars = encoded_df.columns.tolist()
explanatory_vars.remove('Mean_Involvement')

# Calculate the VIF values
X = encoded_df[explanatory_vars].astype(float)
vif = pd.DataFrame()
vif["Variable"] = X.columns
vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

# remove the age and regular virtual meetings columns as their VIF score is higher than 10
X = X[['GENDER_Male', 'Demographic_middle', 'Demographic_older',
       'Demographic_parent', 'Demographic_student',
       "online_meetings_experience_I've had online meetings before"]]
vif = pd.DataFrame()
vif["Variable"] = X.columns
vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

from sklearn.model_selection import train_test_split

y = encoded_df['Mean_Involvement']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)



