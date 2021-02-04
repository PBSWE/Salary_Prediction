#!/usr/bin/env python
# coding: utf-8

# # Salary Predictions Project - Machine Learning

# # Part 1 - DEFINE

# ### ---- 1 Define the problem ----

# We want to be able to create salary predictions of certain jobs based on the dataset provided that cointains info about job titles, distance from city, years of experience and other attritbutes. 

# In[8]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn import preprocessing, metrics
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV, RandomizedSearchCV,RepeatedStratifiedKFold
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier
import inspect
import xgboost
import pickle

get_ipython().run_line_magic('matplotlib', 'inline')

#Author
__author__ = "Phil Baltazar"
__email__ = "phillusnow@gmail.com"
__website__ = "https://github.com/pbswe"


# ## Part 2 - DISCOVER

# ### ---- 2 Load the data ----

# In[9]:


trainFeatDF = pd.read_csv('../Salary_Prediction_Portfolio/data/train_features.csv') 
trainSalsDF = pd.read_csv('../Salary_Prediction_Portfolio/data/train_salaries.csv')
testFeatDF = pd.read_csv('../Salary_Prediction_Portfolio/data/test_features.csv')


# ### ---- 3 Clean the data ----

# In[10]:


# Briefly examine the data.

trainFeatDF.head(10)


# In[11]:


trainSalsDF.head(10)


# In[12]:


testFeatDF.head(10)


# In[13]:


print(trainFeatDF.shape)
print(testFeatDF.shape)
print(trainSalsDF.shape)


# In[14]:


# Verify dtype and length of each dataset via .info() method.


# In[15]:


trainFeatDF.info()


# In[16]:


trainSalsDF.info()


# In[17]:


testFeatDF.info()


# In[18]:


# Identify numerical and categorical variables.


# In[19]:


trainFeatDF.columns


# In[20]:


# trainFeatDF has 8 columns: 2 of type "int64" and 6 of type "object".
# trainSalsDF has 2 columns: 1 of type "int64" and 1 of type "object".
# testFeatDF has 8 columns:  2 of type "int64" and 6 of type "object".

# Check for missing data and add "NaN" if any found. 


# In[21]:


trainFeatDF.replace('?', np.nan, inplace=True)


# In[22]:


trainSalsDF.replace('?', np.nan, inplace=True)


# In[23]:


testFeatDF.replace('?', np.nan, inplace=True)


# In[24]:


# Look for duplicate data, invalid data (e.g. salaries <=0), or corrupt data and remove it.


# In[25]:


trainFeatDF.duplicated().sum()


# In[26]:


trainSalsDF.duplicated().sum()


# In[27]:


testFeatDF.duplicated().sum()


# In[28]:


# Separate both variable types and summarize them.


# In[29]:


trainFeatDF.describe(include=[np.number])


# In[30]:


trainFeatDF.describe(include=['O'])


# In[31]:


numericCols = ['yearsExperience', 'milesFromMetropolis']


# In[32]:


categoricCols = ['jobId', 'companyId', 'jobType', 'degree', 'major', 'industry']


# In[33]:


# Merge both train_features and train_salaries into one dataframe. Delete the original DFs.


# In[34]:


trainDF = pd.merge(trainFeatDF, trainSalsDF, on='jobId')


# In[35]:


# Before deleting the original DFs, check that the new one is correct.
trainDF.head()


# In[36]:


# Deleting the previous DFs to save memory. 
del trainFeatDF
del trainSalsDF


# In[37]:


trainDF['companyId'].value_counts()


# In[38]:


# Changing salary type to float, is a better representation for currency.

trainDF['salary'] = trainDF['salary'].astype(float)

trainDF.info()
trainDF['salary'].head


# ### ---- 4 Explore the data (EDA) ----

# Visualization of target variable (salary);
# 
# Summarize each feature/target variable;
# 
# Look for correlation between each feature and the target.

# In[39]:


plt.figure(figsize = (12, 6))
plt.subplot(1, 2, 1)
sns.boxplot(trainDF.salary)
plt.subplot(1, 2, 2)
sns.distplot(trainDF.salary, bins=20)
plt.show()


# Using IQR rule to identify potential outliers.

# In[40]:


stat = trainDF.salary.describe()
print(stat)
IQR = stat['75%'] - stat['25%']
print(IQR)
upper = stat['75%'] + 1.5 * IQR
lower = stat['25%'] - 1.5 * IQR
print('The upper and lower bounds for the suspected bounds are {} and {} respectively.'.format(upper, lower))


# In[41]:


# Checking for potential outliers below the lower bound.
trainDF[trainDF.salary < 8.5]
trainDF.loc[trainDF.salary < lower]


# In[42]:


trainDF.shape


# In[43]:


# Dropping these lower (zero) salaries since they don't add anything to our model.
trainDF = trainDF[trainDF['salary'] > lower]


# In[44]:


# Double checking to see if rows above were dropped.
trainDF.shape


# In[45]:


# Checking for potential outliers above the upper bound. 

trainDF.loc[trainDF.salary > 222.5, 'jobType'].value_counts()


# In[46]:


# Checking for the most suspicious outliers above upper bound.

trainDF[(trainDF.salary > 222.5) & (trainDF.jobType == 'JUNIOR')]


# In[47]:


# Junior salaries over the 75% percentile are rare (16 out of 1mil), but they could occur. 
# Therefore, these outliers are acceptible. 


# In[48]:


# Value_counts gives you how many features of a column from a variable of type object.
trainDF['jobType'].value_counts()


# In[49]:


# Visualize correlation between variables of type number.

trainDF[['yearsExperience', 'milesFromMetropolis', 'salary']].corr()


# In[50]:


# Function that creates plots to eplore the feature variables
def plotFeats(trainDF, var): 
    '''
    produce plot for each features
    plot1(left), the distribution of samples on the feature
    plot2(right), the dependance of salary on the feature
    '''
    plt.figure(figsize = (17, 4))
    plt.subplot(1, 2, 1)
    if trainDF[var].dtype == 'int64':
        plt.hist(trainDF[var], bins=5)
    else:
        # change the object datatype of each variable to category /n
        # type and order their level by the mean salary
        mean = trainDF.groupby(var)["salary"].mean()
        trainDF[var] = trainDF[var].astype('category')
        level = mean.sort_values().index.tolist()
        trainDF[var].cat.reorder_categories(level, inplace=True)
        trainDF[var].value_counts().plot(kind='bar')   
    plt.xticks(rotation=45, size=8)
    plt.xlabel(var)
    plt.ylabel('Count')
    
    plt.subplot(1, 2, 2)
    # There are many different companyIds so its better to visualize with a line graph
    if trainDF[var].dtype == 'int64' or var == "companyId": 
        # Plot the mean salary for each category and shade the line /n
        # between the (mean - std, mean + std)
        mean = trainDF.groupby(var)["salary"].mean()
        std = trainDF.groupby(var)["salary"].std()
        mean.plot()
        plt.fill_between(range(len(std.index)), mean.values-std.values, 
                         mean.values + std.values,alpha = 0.1)
    else:
        sns.boxplot(x= var, y="salary", data= trainDF)
    
    plt.xticks(rotation=45)
    plt.ylabel('Salary')
    plt.show()


# In[51]:


# The below code was used to sort and plot prior to completing the function above.
# It is now deprecated as long as the function above is in use.

#sorted_list = trainDF.groupby('jobType')['salary'].median().sort_values(ascending=False).index
#boxPlotGraph = sns.boxplot(x="jobType", y="salary", data=trainDF, order=sorted_list)
#plt.xticks(rotation=60)
#plt.figure(figsize = (20, 20))


# In[52]:


plotFeats(trainDF, 'jobType')


# In[53]:


plotFeats(trainDF, 'companyId')


# In[54]:


# Company Id looks a bit cluttered, but it's still visible that it's a poor predictor of salary, /n
# since it shows a straight horizontal line across all features.
# Here's another companyId visualization.

boxPlotGraph = sns.boxplot(x="companyId", y="salary", data=trainDF, order=None)
plt.xticks(rotation=60)
plt.figure(figsize = (20, 20))


# In[55]:


plotFeats(trainDF, 'degree')


# In[56]:


plotFeats(trainDF, 'major')


# In[57]:


plotFeats(trainDF, 'industry')


# In[58]:


plotFeats(trainDF, 'yearsExperience')


# In[59]:


# Years of experience has a nice, clear correlation with salary. /n
# It's a progressive line showing the higher the years of experience, the higher the salary.

boxPlotGraph = sns.boxplot(x="yearsExperience", y="salary", data=trainDF)
plt.xticks(rotation=60)
plt.figure(figsize = (20, 20))


# In[60]:


plotFeats(trainDF, 'milesFromMetropolis')


# In[61]:


# Looks like the salary mean is around 100-105. 

trainDF.salary.hist() #add or remove ';' at the end for a small difference in output.


# In[62]:


# Encoding categorical data in a dataframe.
def EncodeData(trainDF):    
    for col in trainDF.columns:
        if trainDF[col].dtype.name == 'category':
            le = LabelEncoder()
            trainDF[col] = le.fit_transform(trainDF[col])
    return trainDF


# In[63]:


# Create copy of dataframe and encode the categorical data.
baselineDF = trainDF.copy()
baselineDF = EncodeData(baselineDF)


# In[64]:


# Plot Seaborn heatmap to visualize correlation between variables.
plt.figure(figsize = (11, 8))
corr = baselineDF.corr()
sns.heatmap(corr,
            xticklabels = corr.columns.values,
            yticklabels = corr.columns.values,
            cmap = "Greens", vmin=-1, vmax=1, annot=True, linewidths=1)
plt.title('Heatmap of Correlation Matrix')
plt.xticks(rotation=45)
plt.show()


# ###### Based on the plotted heatmap, the strongest correlators are YoE and miles from metropolis.
# <br>
# We also see no collinearity between features, as well as companyId having near zero correlation with salary, so we could consider dropping it after the baseline model executes.

# ### ---- 5 Establish a baseline ----

# In[65]:


# Selecting a reasonable metric, using "average salary" for each industry \n
# as the base and then measure MSE during 5-fold cross-validation.


# In[66]:


baselineDF.columns


# In[67]:


# Splitting features and targets
featuresBaseline = baselineDF[['companyId', 'jobType', 'degree', 'major', 'industry',
                            'yearsExperience', 'milesFromMetropolis', 'salary']]
targetsBaseline = baselineDF[['salary']]
# (optional)
# del baselineDF


# In[68]:


# Creating an extremely simple model and measure its efficacy.
lr = LinearRegression()
baselineLR_mse = cross_val_score(lr, featuresBaseline, targetsBaseline, scoring 
                                  = 'neg_mean_squared_error')
baselineLR_mse = -1*baselineLR_mse.mean()
print("Baseline MSE Score: ", baselineLR_mse)


# ### ---- 6 Hypothesize solution ----

# Brainstorming a few models that may improve results over the baseline model based on the EDA. We'll see which one performs best and pick that one to fine tune it.

# ## Part 3 - DEVELOP

# #### Metric will be MSE and the goal is:
#  - <360 for entry-level data science roles
#  - <320 for senior data science roles

# ### ---- 7 Engineer features  ----

# In[69]:


# Checking that the data is ready for modeling, and create any \n
# new features needed to potentially enhance the model.


# In[70]:


# One hot encoding, create new features if needed. Tune models as shown below.

categoryDF = trainDF[['jobType', 'degree', 'major', 'industry']]
categoryDF = pd.get_dummies(categoryDF, drop_first=True)


# In[71]:


categoryDF.shape


# In[72]:


# Normalizing values between 0 and 1 using Min/Max Scaler.

normalDF = trainDF[['yearsExperience', 'milesFromMetropolis']]
cols = normalDF.columns
normalDF = MinMaxScaler().fit_transform(normalDF)
normalDF = pd.DataFrame(normalDF, columns = cols)


# In[73]:


# Merging (concat) the converted categorical and numerical variables.

featuresDF = pd.concat([categoryDF.reset_index(drop=True), normalDF], axis=1)
targetsDF = trainDF[['salary']]
#del categoryDF, normalDF


# In[74]:


featuresDF.shape


# ### ---- 8 Create models ----

# In[75]:


# Developing and tuning the models brainstormed during part 2.


# In[76]:


# Utilizing 5 Fold - Cross Validation (CV) of each model.

#'''In this case, MSE (mean squared error) is the best metric to measure the efficacy /n
#    because the prediction here is on salaries, which are numerical in nature. '''
# groupy for each industry and get the error from that.

def evalModel(model):
    negMse = cross_val_score(model, featuresDF, targetsDF.values.ravel(), scoring
                            = 'neg_mean_squared_error')
    mse = -1 * negMse
    stdMse = round(mse.std(), 2)
    meanMse = round(mse.mean(), 2)
    print('\nModel:\n', model)
    print('    Standard Deviation of Cross Validation MSEs:\n     ', stdMse)
    print('    Mean 5-Fold Cross Validation MSE: \n      ', meanMse)
    return meanMse


# ### ---- 9 Test models ----

# In[77]:


# Running 5-fold cross validation on models and measuring MSE.


# ###### Warning.

# ###### Warning!

# ###### WARNING!
# 
# Friendly reminder: the next cells take 30-60+ minutes (each) to run. Make sure to save this notebook and push/commit to Git for version control as a precautionary step. :)

# In[79]:


# Looping through different models to obtain their MSE. Hyperparameters chosen manually (testing).

models = []
meanMse = {}

lr = LinearRegression()

sgd = SGDRegressor(max_iter=200, learning_rate='optimal')

dtr = DecisionTreeRegressor(max_depth=15)

rfr = RandomForestRegressor(n_estimators=150, n_jobs=-1, max_depth=30, 
                            min_samples_split=60, max_features='sqrt')
xgb = xgboost.XGBRegressor(max_depth=5, n_estimators=500, learning_rate=0.1, n_jobs=-1)

models.extend([lr, sgd, dtr, rfr, xgb])

print('Cross Validation of Models Initiated...\n')

for model in models:
    mseIter = evalModel(model)
    meanMse.update({model:mseIter})
    
bestModel = min(meanMse, key=meanMse.get)

print('\n\nThe model with the lowest average MSE to use for predictions is:\n')
print(bestModel)


# In[80]:


# Creating a pandas dataframe from the meanMse dictionary.
modelsDF = pd.DataFrame.from_dict(data = meanMse, orient='index', columns=['MSE-Score'])
modelsDF.index = ['LR', 'SDG', 'DTR', 'RF', 'XGB']
modelsDF


# In[81]:


# Visualizing the accuracy results.
modelsDF.plot(kind='bar')
plt.xticks(rotation=45)


# ### ---- 10 Select best model  ----

# ###### Selecting the model with the lowest error as the "production" model \n
# As we see above, XGB has the smallest MSE. Now, we'll fine tune it. 
# 

# In[82]:


trainX, testX, trainY, testY = train_test_split(featuresDF, targetsDF, 
                                                random_state=36, test_size=0.2)


# In[83]:


# Creating an array below to store the results, and a watchlist to keep track of performance.
results = []

eval_set = [(trainX, trainY), (testX, testY)]


# In[84]:


# Checking hyperparameters.

print(inspect.signature(xgboost.XGBRegressor))


# In[85]:


results = [] # resets it.

# Enumerating through different n_estimators values and storing results.

for n_estimators in [100, 150, 250, 500, 750, 1000]:
    clf = xgboost.XGBRegressor(max_depth=5, learning_rate=0.1, n_estimators
                               =n_estimators, n_jobs=-1)
    clf.fit(trainX, trainY, eval_set = eval_set, verbose=False)
    results.append(
        {
            'n_estimators': n_estimators, 
            'train_error': metrics.mean_squared_error(trainY, clf.predict(trainX)),
            'test_error': metrics.mean_squared_error(testY, clf.predict(testX))
        })

# Showing results
nEstimatorsLr = pd.DataFrame(results).set_index('n_estimators').sort_index()
nEstimatorsLr


# In[193]:


# Visualizing the n_estimator learning curve. 
nEstimatorsLr.plot(title='nEstimator Learning Curve')


# ###### What does this Learning Curve tells us?
# The train error tends to go to zero as it exhausts all features in the dataset. The test however, plateaus at about 500, so optimally that's what we should use.

# In[194]:


results = [] # resets it again.

# Enumerating different max_depth values and storing results.

for max_depth in [3, 4, 5, 6, 8, 10]:
    clf = xgboost.XGBRegressor(max_depth = max_depth, n_estimators = 750, learning_rate 
                              = 0.1, n_jobs=-1)
    clf.fit(trainX, trainY, eval_set = eval_set, verbose = False)
    results.append(
        {
            'max_depth': max_depth,
            'train_error': metrics.mean_squared_error(trainY, clf.predict(trainX)),
            'test_error': metrics.mean_squared_error(testY, clf.predict(testX))
        })

# Displaying results.
maxDepthLr = pd.DataFrame(results).set_index('max_depth').sort_index()
maxDepthLr


# In[195]:


# Visualizing the maxDepth learning curve. 
maxDepthLr.plot(title='maxDepth Learning Curve')


# ###### What does the Max Depth Learning Curve tells us?
# The optimal value for maxDepth is around 4 and n_estimators is 750, while learning_rate is 0.1.
# After that point we see the test error curve increase. 

# In[196]:


results = [] # resetting the array again. 

# Enumerating through different learning_rate values and storing results.
for learning_rate in [0.05, 0.1, 0.2, 0.3]:
    clf = xgboost.XGBRegressor(max_depth=4, learning_rate=learning_rate, n_estimators=750, n_jobs=-1)
    clf.fit(trainX, trainY, eval_set=eval_set, verbose=False)
    results.append(
        {
            'learning_rate': learning_rate,
            'train_error': metrics.mean_squared_error(trainY, clf.predict(trainX)),
            'test_error': metrics.mean_squared_error(testY, clf.predict(testX))
        })
    
# Displaying Results  
learningRateLr = pd.DataFrame(results).set_index('learning_rate').sort_index()
learningRateLr


# In[197]:


# Visualizing the Learnin Rate learning curve. 
learningRateLr.plot(title='Learning Rate Learning Curve')


# ###### What does the Learning Rate learning curve tells us?
# The optimal learning_rate value could be between 0.05 and 0.10 while max_depth=4 and n_estimators=750. The model with learning_rate=0.05 may have been unable to converge, therefore we will explore early stopping with this learning rate.

# In[198]:


# Saving the best hyperparameter setting.
bestModel = xgboost.XGBRegressor(learning_rate=0.05, max_depth=4, n_estimators=2000, n_jobs=-1)


# In[86]:


# Fitting the training set and applying early stopping.
bestModel.fit(trainX, trainY, early_stopping_rounds=10, eval_set=eval_set, verbose=False)


# In[200]:


# Saving the best hyperparameter setting with n_estimators from early stopping.
bestModel = xgboost.XGBRegressor(learning_rate=0.05, max_depth=4, n_estimators=1500, n_jobs=-1)


# In[202]:


# Running the model in the same way as the intial test to see improvement of MSE.
evalModel(bestModel)


# ###### Conclusion
# While the model has not improved by much after some fine-tuning, these new hyperparameters will still be utilized for deployment.

# ## Part 4 - DEPLOY

# ### ---- 11 Automate pipeline ----

# In[203]:


#write script that trains model on entire training set, saves model to disk,
#and scores the "test" dataset


# In[204]:


# Creating a function that trains model on entire training set and saves the model to disk.

def trainedCleanModelDF(raw_train_features, raw_train_targets):
    
    # Loading csv files
    featuresDF = pd.read_csv(raw_train_features)
    targetsDF = pd.read_csv(raw_train_targets) 

    # Cleaning feature and target dataframes per analysis above
    df = pd.merge(featuresDF, targetsDF, on='jobId')
    df = df[df['salary'] > 8.5]
    categoriesDF = df[['jobType', 'degree', 'major', 'industry']]
    categoriesDF = pd.get_dummies(categoriesDF, drop_first=True)
    featuresDF = pd.concat([categoriesDF, df[['yearsExperience', 'milesFromMetropolis']]], axis=1)
    targetsDF = df[['salary']]
    del categoriesDF, df
    
    # Implement best model discovered per analysis above
    model = xgboost.XGBRegressor(learning_rate=0.05, max_depth=4, n_estimators=1500)
    model.fit(featuresDF, targetsDF)
    
    # Save model to disk
    model_file = 'model'
    pickle.dump(model, open(model_file, 'wb'))
    
    # Informs user that process is complete
    print("Data prepraration and model creation complete.")


# In[205]:


# Script that prepares data, predicts salaries, and exports results
# This class will be saved in a .py file for private use - See "Salary_Prediction_Module.py"
class salaryPredictionModel():
    
    # Read the 'model' file which was saved
    def __init__(self, model_file):
        self.xgb = pickle.load(open(model_file, 'rb'))
    
    # Takes data, prepares data, makes predictions from trained model, and exports results to csv file
    def exportPredictions(self, data_file):

        # Load csv file
        df_pred_features = pd.read_csv(data_file)
    
        # Saves jobId column for output file
        df_pred_jobId = pd.DataFrame(df_pred_features['jobId'])
    
        # Prepares data to be fed into the model
        df_pred_categories = df_pred_features[['jobType', 'degree', 'major', 'industry']]
        df_pred_categories = pd.get_dummies(df_pred_categories, drop_first=True)
        df_pred_features = pd.concat([df_pred_categories, df_pred_features[['yearsExperience','milesFromMetropolis']]], axis=1)
        del df_pred_categories
    
        # Loads model from disk, predicts salaries, and exports results to .csv file
        df_pred = pd.DataFrame(self.xgb.predict(df_pred_features))
        df_pred.columns = ['salary']
        df_pred = pd.concat([df_pred_jobId,df_pred], axis=1)
        df_pred.to_csv('predicted_salaries.csv')
        del df_pred_jobId
        
        # Informs user that process is complete
        print("Predictions exported to .csv file.")
    
    # Plot feature importance of model and save figure to .jpg file
    def exportFeatureImportance(self):
        fig, ax = plt.subplots(figsize=(12, 8))
        xgboost.plot_importance(self.xgb, height=0.6, ax=ax)
        fig.savefig('feature_importance.jpg')
    
        # Informs user that process is complete
        print("Feature importances exported to .jpg file.")


# ### ---- 12 Deploy solution ----

# In[207]:


# Use pipeline to prepare data, fit to model, and save to disk
trainedCleanModelDF("data/train_features.csv", "data/train_salaries.csv")


# In[209]:


# Load and initialize the saved model
model = salaryPredictionModel('model')


# In[211]:


# Prepare new data and export predictions to .csv file
model.exportPredictions("data/test_features.csv")


# In[213]:


# Export feature importance chart
model.exportFeatureImportance()


# ### ---- Conclusion ----
# When predicting salaries from datasets such as the ones we used, the strongest correlation features are as shown above, with miles from metropolis and years of experience being the top two. 
# 
# <br>
# 
# The Tableau graphs also show a linear progression as years of experience increases. It also shows the progression from each level of experience. 
# 
# <br>
# 
# In other words, the Tableau graph has for each year of experience, ranging from 0 to 24, a cluster of what each level for that particular amount experience represents, starting with Janitor being the lowest paid and CEO being the highest paid, per year of experience. 

# In[ ]:




