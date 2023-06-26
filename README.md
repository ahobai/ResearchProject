# ResearchProject
The influence of individual backgrounds on conversational involvement in a group setting. This experiment is part of the [Research Project 2023](https://github.com/TU-Delft-CSE/Research-Project).


## Dependencies used
This project uses the packages enumerated below which can be installed by running the commands next to them in the anaconda environment shell:
- pandas (for using data frames to process the CSV files): 'pip install pandas'
- numpy (for arrays and other mathematical usages): 'pip install numpy'
- matplotlib (for graphs and plots): 'pip install matplotlib'
- seaborn (for data visualizations): 'pip install seaborn'
- statsmodels (for statistical operations and the GLMM): 'pip install statsmodel'
- sklearn (for LR, DT, and RF models): 'pip install sklearn'
- researchpy (for data analysation): 'pip install researchpy'

## Files description
1. Modelling.py file contains the final experiment (explanatory variables selection, processing the target variable, data visualization, and data modelling based on these variables - using Linear Regression, Decision Tree, Random Forest and Mixed Effects Model for k-fold cross-validation method and Linear Regression, Decision Tree, Random Forest, Mixed Effects, Gaussian Process Regression and GPBoost Models for a normal split of data; the models' performances from the k-fold cross-validation results are stored in the performancedf variable, whereas the ones from the normal split are stored in the results data frame)
2. Target_variable.py file contains information about the processing and assessment of the target variable (the inter-rater reliability methods used, as well as how it was created from the four annotations sets)
3. Explanatory_variable.py file contains data about the explanatory variable (how it was preprocessed and the visualizations of the predictors)
