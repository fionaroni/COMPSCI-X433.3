Team4DASC = 'Fiona Tang AND Zhicheng (Jason) Xue'
print('Team members for X433.3 Python for Data Analysis and Scientific Computing are %s' %Team4DASC )
Team4DASCemail = 'fionatang@berkeley.edu; emailxjason@gmail.com'
print('\nOur emails are %s' %Team4DASCemail )

# ## Import all libraries needed for the project
import numpy as np
import scipy.stats as stats
from scipy.stats import normaltest, anderson, shapiro
import pandas as pd
import matplotlib.pyplot as plt
import plotly
plotly.offline.init_notebook_mode()
import seaborn as sns
import tensorflow as tf
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.preprocessing import normalize
from tensorflow.contrib.factorization import KMeans
np.random.seed(2018)

# =============================================================================
# ## Load csv data into pandas dataframe
# =============================================================================
df_full = pd.read_csv('loan.csv')
# ### Check columns 19 and 55 based on warning message
column_list=df_full.columns.tolist()
print('There are', len(column_list), 'columns in the dataframe')
print('Col 19 is', column_list[19], 'and data type is', df_full[column_list[19]].dtypes)
print('Col 55 is', column_list[55], 'and data type is', df_full[column_list[55]].dtypes)
print(df_full.iloc[:,19:20].head(10))
print(df_full.iloc[:,55:56].head(10))
print(df_full.verification_status_joint.unique())
# #### Comment: both columns are string columns with NaN used for missing values and that is reason the warning showed up during data import step

# =============================================================================
# ## Check all available data types in the full dataframe
# =============================================================================
df_full.dtypes.unique()

# ### a.Check columns whose data type is int64
columns_int64=df_full.select_dtypes(include='int64').columns.tolist()
print('Number of int64 columns are: ', len(columns_int64))
print('int64 columns are: ',columns_int64)
# ####  Comment: id and member_id columns will not be useful for this analysis, therefore both will be removed from the dataframe
unwanted = ['id', 'member_id']
for i in unwanted:
    df_full.drop(i, axis=1, inplace=True)
    
# ### b. Check columns whose data type is float64
columns_float64 = df_full.select_dtypes(include='float64').columns.tolist()
print('Number of float64 columns are: ', len(columns_float64))
print('float64 columns are: ',columns_float64)
columns_float64_na_pct={}
for col in columns_float64: 
    columns_float64_na_pct[col]=np.sum(np.isnan(df_full[col]))/np.size(df_full[col]) # key is col, value is percentage of nan
columns_float64_na_drop=[]
for key, value in columns_float64_na_pct.items():
    if columns_float64_na_pct[key]>=0.95: # if percentage of nan is greater than 95%
        columns_float64_na_drop.append(key) # append the column to a dict called columns_float64_na_drop
print(columns_float64_na_drop)
columns_float64_keep = [x for x in columns_float64 if (x not in columns_float64_na_drop)]
print('Float64 columns to keep:',columns_float64_keep)
# #### Comment: During this step, float64 columns with high percentage of missing values were dropped as they add little value to the analysis afterwards

# ### c. Check columns whose data type is string
columns_string=df_full.select_dtypes(include='O').columns.tolist()
print('Number of string columns are: ', len(columns_string))
print('string columns are: ',columns_string)
columns_string_na_drop=[] # columns that should be dropped due to high % of NAs
columns_string_other_drop=[] # columns that should be dropped due to large number of distinct values or other reasons
for col in columns_string:
    if df_full[col].isnull().sum(axis=0)/np.size(df_full[col])>=0.95:
        columns_string_na_drop.append(col)
    elif (len(df_full[col].unique())>=100) | ('_d' in col):
        columns_string_other_drop.append(col)
print('Columns that should be dropped due to high % of NAs:', columns_string_na_drop)
print('Columns that should be dropped due to other reasons:', columns_string_other_drop)
columns_string_keep_temp = [x for x in columns_string if (x not in columns_string_na_drop)&(x not in columns_string_other_drop)]
print(columns_string_keep_temp)

# #### Comment: string columns with high percentage of missing values or other reasons such as large number of distinct values were dropped due to the little value and complexity they can add to the analysis afterwards
for col in columns_string_keep_temp:
    print('String column name is:',col,'with number of unique values',len(df_full[col].unique()))
    print(df_full[col].isnull().sum(axis=0))
    print(df_full[col].value_counts())
    
# =============================================================================
# ## Generate target variable based on loan status for developing machine learning model
# =============================================================================
def CreateArrays(col):
    values = df_full[col].value_counts().index.values
    counts = df_full[col].value_counts().values
    return values,counts
def targetFunc(var):
    if var in np.array(['Default','Late (16-30 days)','Late (31-120 days)','Charged Off']):
        target = 1
    elif var in np.array(['Current']):
        target = 0
    else:
        target = -1
    return target
df_full['target'] = df_full['loan_status'].apply(lambda x: targetFunc(x))
def targetLabelFunc(var):
    if var in np.array(['Default','Late (16-30 days)','Late (31-120 days)','Charged Off']):
        target_label = 'Default/Late/ChargeOff'
    elif var in np.array(['Current']):
        target_label = 'Current'
    else:
        target_label = 'Others'
    return target_label
df_full['target_label'] = df_full['loan_status'].apply(lambda x: targetLabelFunc(x))
df_full['loan_status'].value_counts()
loan_status_values, loan_status_counts = CreateArrays('loan_status')
print('loan_status_values:', loan_status_values)
target_label = np.array(['Current', 'Others', 'Default/Late/ChargeOff'])
print(target_label)
target_values, target_counts = CreateArrays('target')
print(target_values);print(target_counts)
plt.figure('Distribution of Target')
plt.axes([0.035, 0.035, 0.9, 0.9])
c = ['green', 'orange', 'red']
e = [0, 0, 0.05]
plt.cla()
plt.pie(target_counts, explode = e, labels = target_label, 
        colors = c, radius = .75, autopct='%1.2f%%', shadow = True, startangle = 15)
plt.axis('equal')
plt.xticks(()); plt.yticks(())
plt.title('Distribution of Target')
plt.show()
# ### Comments: 2/3 of the sample are current on their loan payment and about 7% of the loans are in some type of default/late payment/charge off status. This is an unbalanced dataset for supervised training.
# #### Comment: In all of the following analysis, I will exclude "Others"
df = df_full[df_full['target']!=-1].copy()
print(columns_float64_keep)

# =============================================================================
# ## SCIPY NORMALITY TESTING and T-TEST AFTER NORMALIZED
# =============================================================================

"""
Much of the field of statistics is concerned with data that assumes that it
was drawn from a Gaussian distribution. If statistical tests are used that
assume Gaussian when your data was drawn from a different distribution, your
findings may be misleading/wrong. There are a number of techniques that you 
can use to check if your data sample is sufficiently Gaussian. If it is not
sufficiently Gaussian, you can instead use non-parametric methods.
"""

lamount_g = df[df['target_label'] == 'Current']['loan_amnt']
lamount_b = df[df['target_label'] == 'Default/Late/ChargeOff']['loan_amnt']
income_g = df[df['target_label'] == 'Current']['annual_inc']
income_b = df[df['target_label'] == 'Default/Late/ChargeOff']['annual_inc']

# normality tests: Shapiro, D’Agostino’s K^2, Anderson-Darling
def normality_test(test_name, data):
    alpha = 0.05
    p=0
    if test_name == shapiro:
        stat, p = test_name(data)
        print("Results from the shapiro test")
        print('Statistics=%.3f, p=%.3f' % (stat, p))
        if p > alpha:
            print('Sample looks Gaussian (fail to reject H0)')
        else:
            print('Sample does not look Gaussian (reject H0)')
    elif test_name == normaltest:
        stat, p = test_name(data)
        print("Results from the D'Agostino's test")
        print('Statistics=%.3f, p=%.3f' % (stat, p))
        if p > alpha:
            print('Sample looks Gaussian (fail to reject H0)')
        else:
            print('Sample does not look Gaussian (reject H0)')
    else: # anderson
        result = test_name(data)
        print("Results from the Anderson test")
        print('Statistic: %.3f' % result.statistic)
        for i in range(len(result.critical_values)):
            sl, cv = result.significance_level[i], result.critical_values[i]
            if result.statistic < result.critical_values[i]:
                print('%.3f: %.3f, data looks normal (fail to reject H0)' % (sl, cv))
            else:
                print('%.3f: %.3f, data does not look normal (reject H0)' % (sl, cv))
        
# run some normality tests, all of which will fail because they have not been normalized
normality_test(shapiro, lamount_g)
normality_test(normaltest, df['int_rate'])
normality_test(anderson, income_b)

"""
In the era of big data, normality tests on a real-world dataset should always 
reject the null hypothesis. When n gets large enough, even the smallest deviation from
perfect normality will influence the shape of distribution. Abd every real-world 
dataset has some degree of randomness.
"""

# however, we can normalize using sklearn
def normalizer(pd_series):
    """sklearn.preprocessing.normalize requires object to be an ndarray, this 
    function converts pandas series to ndarray, normalizes the array, and 
    returns the normalized array"""
    arr = np.array(pd_series) # convert from pandas series to array
    ndarr = arr.reshape(-1,1) # reshape to 2 dimensional array
    arr_normed = normalize(ndarr) # normalize
    return arr_normed

# normalize 
lamount_g_normed = normalizer(lamount_g)
lamount_b_normed = normalizer(lamount_b)
income_b_normed = normalizer(income_b)
income_g_normed = normalizer(income_g)
# running normality tests again will show that 'Sample looks Gaussian'
normality_test(normaltest, income_b_normed)
normality_test(normaltest, lamount_g_normed)

# now we can test the difference in means of incomes, good vs. bad
scipy.stats.ttest_ind(income_g_normed, income_b_normed, nan_policy='omit')
# output shows Ttest_indResult(statistic=array([-0.4480938]), pvalue=array([0.65408565]))\

# =============================================================================
# ## Exploratory Analysis on Numerical Variables
# =============================================================================
# ### Annual income
sns.distplot(df.annual_inc[df.annual_inc<df.annual_inc.quantile(0.9)], hist=True, kde=True, 
             bins=50, color = 'blue',
             hist_kws={'edgecolor':'black'})
plt.title('Histogram of Annual Income')
plt.xlabel('Income')
plt.xticks([0, 20000, 40000, 60000, 80000, 100000, 120000],['$0', '$20k', '$40k', '$60k', '$80k', '$100k', '$120k'])
plt.ylabel('Density')
df.annual_inc.describe()
bins_annual_inc = [0, 45000, 65000, 90000, 150000, 300000, 500000, 1000000]
labels_annual_inc = ['A_LT45k','B_45kTO65K', 'C_65KTO90K', 'D_90KTO150K', 'E_150KTO300K', 'F_300KTO500K', 'G_Above1M']
df['annual_inc_binned'] = pd.cut(df['annual_inc'], bins=bins_annual_inc, labels=labels_annual_inc)
annual_inc_df = pd.DataFrame(df.pivot_table(values='target',index=['annual_inc_binned'],aggfunc=lambda x: len(x),dropna=True))
annual_inc_df = annual_inc_df.rename(columns={'target': 'NumberOfLoans'})
annual_inc_df.reset_index(level=0, inplace=True)
annual_inc_df.drop([7],inplace=True)
annual_inc_df = annual_inc_df.sort_values('annual_inc_binned')
annual_inc_df_bad = pd.DataFrame(df.pivot_table(values='target',index=['annual_inc_binned'],aggfunc=lambda x: np.mean(x),dropna=True))
annual_inc_df_bad = annual_inc_df_bad.rename(columns={'target': 'Percentage of Bad Loans'})
annual_inc_df_bad.reset_index(level=0, inplace=True)
annual_inc_df_bad.drop([7],inplace=True)
annual_inc_df_bad = annual_inc_df_bad.sort_values('annual_inc_binned')
annual_inc_df_final = pd.merge(annual_inc_df, annual_inc_df_bad,  how='left', left_on=['annual_inc_binned'], right_on = ['annual_inc_binned'])
annual_inc_df_final
plt.show()
plt.close()
# ### comment: There is a negative correlation between annual income and chance of default, which matches our intuition. Higher income people are less likely to default on their loans

# ### DTI
sns.distplot(df.dti[df.dti<df.dti.quantile(0.9)], hist=True, kde=True, 
             bins=50, color = 'red',
             hist_kws={'edgecolor':'black'})
plt.title('Histogram of Debt to Income Ratio')
plt.xlabel('DTI')
plt.ylabel('Density')
plt.show()
plt.close()
df[df.dti != 9999].dti.describe()
# ### comment: the distribution of DTI is right skewed with a long tail. Based on the distribution, the data should represent percentage of income (DTI=18.75 represents 18.75% of income is tied with debts).

# ### Relationship between Income and Laon Status
# overlaid probability density functions
fig, ax = plt.subplots(figsize=(8,4))
bins=np.linspace(0, 300000, 25)
plt.hist(income_g, bins, density=True, color='green', alpha=0.1, histtype='stepfilled', label='Current')
plt.hist(income_b, bins, density=True, color='red', alpha=0.1, histtype='stepfilled', label='Default/Late/Chargeoff')
ax.legend(loc='upper right')
plt.title('Probability Density Function\nAnnual Incomes: Current vs. Default/Late/Chargeoff')
plt.xlabel('Annual Income')
plt.ylabel('Density')
plt.xticks([0, 50000, 100000, 150000, 200000, 250000, 300000],['$0', '$50k', '$100k', '$150k', '$200k', '$250k', '$300k'], rotation=90)
plt.yticks(rotation=20)
plt.grid(True)
plt.show()
# the population of borrowers with bad loan status have lower mean and median annual incomes
print(np.mean(income_g))
print(np.median(income_g)) 
print(np.mean(income_b))
print(np.median(income_b))

# overlaid cumulative density functions:
fig, ax = plt.subplots(figsize=(8,4))
n, bins, patches = ax.hist(income_g, bins, density=True, color='green', alpha=0.1, histtype='stepfilled', cumulative=True, label='Current')
ax.hist(income_b, bins, density=True, color='red', alpha=0.1, histtype='stepfilled', cumulative=True, label='Default/Late/Chargeoff')
ax.legend(loc='lower right')
plt.title('Cumulative Density Function\nAnnual Incomes: Current vs. Default/Late/Chargeoff')
plt.xlabel('Annual Income')
plt.ylabel('Density')
plt.xticks([0, 50000, 100000, 150000, 200000, 250000, 300000],['$0', '$50k', '$100k', '$150k', '$200k', '$250k', '$300k'], rotation=90)
plt.yticks(rotation=20)
plt.grid(True)
plt.show()

# overlaid CDFs - expected  
fig, ax = plt.subplots(figsize=(8,4))
values_g, base = np.histogram(income_g, bins, density=True)
values_b, base = np.histogram(income_b, bins, density=True)
cumulative_g = np.cumsum(values_g) #evaluate the cumulative
cumulative_b = np.cumsum(values_b)
plt.plot(base[:-1], cumulative_g, c='green', label='Current') # plot the cumulative function
plt.plot(base[:-1], cumulative_b, c='red', label='Default/Late/Chargeoff')
ax.annotate('Current has first order \nstochastic dominance', xy=(80000, 0.00006), xytext=(130000, 0.000055), arrowprops=dict(facecolor='green', width=2, shrink=0.05))
ax.legend(loc='lower right')
plt.title('Cumulative Density Function\nAnnual Incomes: Current vs. Default/Late/Chargeoff')
plt.xlabel('Annual Income')
plt.ylabel('Density')
plt.xticks([0, 50000, 100000, 150000, 200000, 250000, 300000],['$0', '$50k', '$100k', '$150k', '$200k', '$250k', '$300k'], rotation=90)
plt.yticks(rotation=20)
plt.grid(True)
plt.show()

"""
Borrowers with good loan status tend to have slightly higher incomes, while 
borrowers with bad loan status tend to have slightly lower incomes.

For any given income amount between $20k and $150k, we observe there is a 
slightly greater proportion of bad loan borrowers that make less than or equal 
to that income amount, as compared to good loan borrowers. Good loan borrowers 
have first order stochastic dominance.
"""

# ### Relationship between Loan Amount & Good vs. Bad Loan Status
## overlaid PDFs: loan amounts of borrowers with good vs. bad loans
fig, ax = plt.subplots(figsize=(8,4))
plt.hist(lamount_g, density=True, color='green', alpha=0.1, histtype='stepfilled', label='Current')
plt.hist(lamount_b, density=True, color='red', alpha=0.1, histtype='stepfilled', label='Default/Late/Chargeoff')
ax.legend(loc='upper right')
plt.title('Probability Density Function\nLoan Amounts: Current vs. Default/Late/Chargeoff')
plt.xlabel('Loan Amount')
plt.ylabel('Density')
plt.xticks([0, 5000, 10000, 15000, 20000, 25000, 30000, 35000],['$0', '$5k', '$10k', '$15k', '$20k', '$25k', '$30k', '$35k'], rotation=20)
plt.yticks(rotation=20)
plt.grid(True)
plt.show()

# overlaid cumulative density functions:
fig, ax = plt.subplots(figsize=(8,4))
n, bins, patches = ax.hist(lamount_g, density=True, color='green', alpha=0.1, histtype='stepfilled', cumulative=True, label='Current')
ax.hist(lamount_b, density=True, color='red', alpha=0.1, histtype='stepfilled', cumulative=True, label='Default/Late/Chargeoff')
ax.legend(loc='lower right')
plt.title('Cumulative Density Function\nLoan Amounts: Current vs. Default/Late/Chargeoff')
plt.xlabel('Loan Amount')
plt.ylabel('Density')
plt.xticks([0, 5000, 10000, 15000, 20000, 25000, 30000, 35000],['$0', '$5k', '$10k', '$15k', '$20k', '$25k', '$30k', '$35k'], rotation=20)
plt.yticks(rotation=20)
plt.grid(True)
plt.show()

# overlaid CDFs - expected  
fig, ax = plt.subplots(figsize=(8,4))
vals_g, base = np.histogram(lamount_g, bins, density=True)
vals_b, base = np.histogram(lamount_b, bins, density=True)
cumul_g = np.cumsum(vals_g) # evaluate the cumulative
cumul_b = np.cumsum(vals_b)
plt.plot(base[:-1], cumul_g, c='green', label='Current') # plot the cumulative function
plt.plot(base[:-1], cumul_b, c='red', label='Default/Late/Chargeoff')
ax.legend(loc='lower right')
plt.title('Cumulative Density Function\nLoan Amounts: Current vs. Default/Late/Chargeoff')
plt.xlabel('Loan Amount')
plt.ylabel('Density')
plt.xticks([0, 5000, 10000, 15000, 20000, 25000, 30000, 35000],['$0', '$5k', '$10k', '$15k', '$20k', '$25k', '$30k', '$35k'], rotation=20)
plt.yticks(rotation=20)
plt.grid(True)
plt.show()

"""
Contrary to what we might expect, borrowers with good loan status tend to take 
out higher loan amounts, while borrowers with bad loan status tend to take out 
lower loan amounts. The difference however is marginal.

Borrowers with good loan status have very, very slight first order stochastic
dominance. We observe that for any given loan amount between $0 and $25k, there
is a slightly lower proportion of good loan borrowers that borrow less than or
equal to that loan amount, as compared to bad loan borrowers. 
"""
# =============================================================================
# ## Exploratory Analysis on Categorical Variables
# =============================================================================
# ### Loan Terms
def CreateArrays(df,col):
    values = df[col].value_counts().index.values
    counts = df[col].value_counts().values
    return values,counts
print(columns_string_keep_temp)
# ### Term
term_values0,term_counts0 = CreateArrays(df[df['target']==0],'term')
term_values1,term_counts1 = CreateArrays(df[df['target']==1],'term')
plt.axes([0.075, 0.075, .88, .88])
p1 = plt.bar(term_values0,term_counts0,color='green')
p2 = plt.bar(term_values1,term_counts1,color='red')
plt.ylabel('Number of Loans/Borrowers')
plt.title('Number of Loans by Term and Target')
plt.legend((p1, p2), ('Current','Default/Late/ChargeOff'))
plt.show()
# ### Comment: More than half of the loans have terms 36 months rather than 60 months

# ### Grade
temp = df.pivot_table(values='target',index=['grade'],aggfunc=lambda x: 100*x.mean())
print ('\nProbility of Not Current for each Credit Grade:') 
print (temp)
fig = plt.figure()
temp.plot(kind = 'bar', title='Probability of Default by Credit Grade',legend=None)
plt.xlabel('Credit Grade')
plt.ylabel('% Probability of Default')
fig.tight_layout()
plt.show()
plt.clf()
plt.close()
# ### Comment: clearly "grade" will be an useful feature for predicting likelihood of default

# ### State
state_list=df.addr_state.value_counts().index.values
print(state_list)
state_df = pd.DataFrame(df.pivot_table(values='target',index=['addr_state'],aggfunc=lambda x: len(x)))
state_df = state_df.rename(columns={'target': 'NumberOfLoans'})
state_df['state'] = state_df.index
state_df.reset_index(level=0, inplace=True)
for col in state_df.columns:
    state_df[col] = state_df[col].astype(str)
scl = [[0.0, 'rgb(242,240,247)'],[0.2, 'rgb(218,218,235)'],[0.4, 'rgb(188,189,220)'],            [0.6, 'rgb(158,154,200)'],[0.8, 'rgb(117,107,177)'],[1.0, 'rgb(84,39,143)']]
data = [ dict(
        type='choropleth',
        colorscale = scl,
        autocolorscale = False,
        locations = state_df['state'],
        z = state_df['NumberOfLoans'].astype(float),
        locationmode = 'USA-states',
        text = state_df['state'],
        marker = dict(
            line = dict (
                color = 'rgb(255,255,255)',
                width = 2
            )
        ),
        colorbar = dict(
            title = "Number of Loans"
        )
    ) ]
layout = dict(
        title = 'Number of Lending Club loans by State<br>(Hover for breakdown)',
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)',
        ),
    )
fig = dict( data=data, layout=layout )
url = plotly.offline.plot( fig, filename='state-cloropleth-map.html' )
# ### Comment: majority of Lending Club's loans are concentrated in big 4 states measured by economy size- California, Texas, Florida and New York.

# ### Home ownership & Employment length
home_emp_df = pd.DataFrame(
    df.pivot_table(
        values='target',index=['emp_length'], columns='home_ownership', 
#         aggfunc=lambda x: (np.mean(x)*100).astype(str) + '%'
#         aggfunc=lambda x: str(((np.round(np.mean(x),decimals=2))*100))+'%'
        aggfunc=lambda x: np.mean(x)
    )
)
print('\nBad Rate of Loans By Home Ownership and Employment Length:') 
home_emp_df.style.format({
    'Any': '{:,.2f}'.format,
    'Mortgage': '{:,.2f}'.format,
    'NONE': '{:,.2f}'.format,
    'OTHER': '{:,.2f}'.format,
    'OWN': '{:,.2f}'.format,
    'RENT': '{:,.2f}'.format,
})
home_emp_df = home_emp_df.reindex(index = [
    '< 1 year', '1 year','2 years', '3 years', '4 years','5 years',  
    '6 years', '7 years', '8 years', '9 years', '10+ years'])
home_emp_df[['RENT','OWN','MORTGAGE','OTHER','ANY','NONE']]
home_emp_df.index
# state_df = state_df.rename(columns={'target': 'NumberOfLoans'})
# home_emp_df['emp_length'] = home_emp_df.index
# home_emp_df.reset_index(level=0, inplace=True)
figu = plt.figure(figsize=(10,6), dpi=120)
ax = figu.add_subplot(111)
plt.plot(home_emp_df.index,home_emp_df['RENT'],color='red', linestyle='--', label='Rent')
plt.plot(home_emp_df.index,home_emp_df['OWN'],color='green', linestyle='-', label='Own')
plt.plot(home_emp_df.index,home_emp_df['MORTGAGE'],color='blue', linestyle='-.', label='Mortgage')
ax.annotate('highest rate \nat 6 years', xy=(6, 0.1322), xytext=(7.1, 0.1305), arrowprops=dict(facecolor='black', width=2, shrink=0.05))
plt.ylabel('Bad Rate')
plt.xlabel('Employment Length')
plt.legend(loc='best')
plt.title('Bad Rate By Home Ownership and Employment Length')
plt.grid()
plt.show()
plt.close()
# ### Comment: Borrowers who rent has higher risk comparing to those who own; those who have about 6 years of employment history tend to show higher risk compared to others

# =============================================================================
# ## MODELING
# =============================================================================
# ## Missing Value Processing
print('Numerical variables to be considered:',columns_float64_keep)
columns_string_keep_temp.remove('loan_status')
print('Categorical variables to be considered:',columns_string_keep_temp)
columns_float64_na_add_dict={}
for var in columns_float64_keep:
    if columns_float64_na_pct[var] > 0:
        columns_float64_na_add_dict[var] = var+'_na'
print(columns_float64_na_add_dict)
for var in columns_string_keep_temp:
    print(var, (df[var].isnull().sum(axis=0))/np.size(df[var]))
columns_string_na_add_dict={}
for var in columns_string_keep_temp:
    if df[var].isnull().sum(axis=0)/np.size(df[var])>0:
        columns_string_na_add_dict[var] = var+'_na'
print(columns_string_na_add_dict)

# ### Create a copy of DF and only keep needed columns
var_list_pre = columns_float64_keep + columns_string_keep_temp
var_list_pre.append('target')
df_final = df[var_list_pre].copy()
def merge_two_dicts(x, y):
    """Given two dicts, merge them into a new dict as a shallow copy."""
    z = x.copy()
    z.update(y)
    return z
na_dict = merge_two_dicts(columns_float64_na_add_dict, columns_string_na_add_dict)
# ### Create dummy variables for columns with NAs to prepare for later stage modeling exercise
print(na_dict)
for key, value in na_dict.items():
    print('key:', key, 'value:', na_dict[key])
    df_final[na_dict[key]] = np.where(df_final[key].isnull(),1,0)
    if key in columns_float64_keep:   
        df_final[key] = np.where(df_final[key].isnull(),0,df_final[key])
    else:
        df_final[key] = np.where(df_final[key].isnull(),'MV',df_final[key])   
    print('Transformation Done!')
    
# ### Mapping categorical variable values to right format with space
for var in columns_string_keep_temp:
    print(var)
    df_final[var].str.strip()
df_final.replace('\s+', '_',regex=True,inplace=True)
df_final['term'].replace('_36_months', '36_months',regex=True,inplace=True)
df_final['term'].replace('_60_months', '60_months',regex=True,inplace=True)
df_final['emp_length'].replace('<_1_year', 'LT_1_year',regex=True,inplace=True)
df_final['emp_length']=np.where(df_final['emp_length']=='10+_years','GT_10_years',df_final['emp_length'])
# ## Dummy variables coding 
df_final = pd.get_dummies(df_final, dummy_na=True,
               columns=[
                        'term', 
                        'grade', 
                        'sub_grade', 
                        'emp_length',
                        'home_ownership', 
                        'verification_status', 
                        'pymnt_plan', 
                        'purpose',
                        'addr_state', 
                        'initial_list_status', 
                        'application_type'                                
])
features = [n for n in df_final.columns.tolist() if n != 'target']
label = ['target']
df_final[features].dtypes.unique()
columns_string_final=df_final[features].select_dtypes(include='O').columns.tolist()
print('Number of string columns are: ', len(columns_string_final))
print('string columns are: ',columns_string_final)

# ## Split dataset into train and test dataset with stratified sampling since this is an unbalanced dataset
# Split dataset into train and test dataset
train_x, test_x, train_y, test_y = train_test_split(df_final[features], df_final[label],
                                                    train_size=0.7, test_size=0.3, stratify=df_final[label],
                                                   random_state=2018)
# Train and Test dataset size details
print("Train_x Shape :: ", train_x.shape)
print("Train_y Shape :: ", train_y.shape)
print("Test_x Shape :: ", test_x.shape)
print("Test_y Shape :: ", test_y.shape)
# ## Data preprocessing
# ### Standardize all features because they were on different scales
scaler = preprocessing.StandardScaler().fit(train_x)
train_x_scaled = scaler.transform(train_x)
test_x_scaled = scaler.transform(test_x)
# ## Train random forest model
rf_model = RandomForestClassifier(n_estimators = 100, class_weight = 'balanced', max_features='sqrt', random_state=2018)
rf_model.fit(train_x_scaled,train_y.values.ravel())
y_pred = rf_model.predict(test_x_scaled)
print('Accuracy:', accuracy_score(test_y,y_pred))
# ### Feature dimension reduction using feature importance output from random forest
feature_imp = pd.Series(
    rf_model.feature_importances_,
    index = train_x.columns).sort_values(ascending=False)
print(feature_imp[feature_imp.values>=0.01])
# Creating a bar plot
sns.barplot(x=feature_imp[feature_imp.values>=0.01], y=feature_imp[feature_imp.values>=0.01].index)
# Add labels to your graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title('Visualizing Important Features')
plt.grid()
plt.show()
tf_features=feature_imp[feature_imp.values>=0.01].index.values
print(tf_features.shape)
train_x.columns
tf_target = 'target'
# ### export train_x, test_x, train_y, test_y to prepare for TensorFlow run
train_x_scaled.shape
train_x.shape
train_y.columns
df_train_feature_xport = pd.DataFrame(train_x_scaled, columns = train_x.columns)[tf_features]
df_train_label_xport   = pd.DataFrame(train_y, columns = train_y.columns)[tf_target]
df_train_xport_all         = df_train_feature_xport.join(df_train_label_xport)
df_test_feature_xport  = pd.DataFrame(test_x_scaled, columns = test_x.columns)[tf_features]
df_test_label_xport    = pd.DataFrame(test_y, columns = test_y.columns)[tf_target]
df_test_xport_all         = df_test_feature_xport.join(df_test_label_xport)
print('Shape of training data for TF:', df_train_xport_all.shape)
print('Shape of test data for TF:', df_test_xport_all.shape)
df_train_xport = df_train_xport_all.sample(frac=1,random_state=2018)
df_test_xport = df_test_xport_all.sample(frac=1,random_state=2018)
print('Shape of training data for TF:', df_train_xport.shape)
print('Shape of test data for TF:', df_test_xport.shape)
# ## TensorFlow Section
tf.reset_default_graph()
# ### Define useful functions for transformations
# Inference used for combining inputs
def combine_inputs(X):
    return tf.matmul(X,W)+b
# new inferred value is the sigmoid applied to the former
def inference(X):
    return tf.sigmoid(combine_inputs(X), name='Inference')
# define sigmoid loss function
def loss(X,Y):
    return tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=combine_inputs(X), labels=Y)
    )
def train(total_loss,learning_rate):
    return tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss)
def evaluate(X,Y):
    predicted=tf.cast(inference(X)>0.5,tf.float32)
    return tf.reduce_mean(tf.cast(tf.equal(predicted,Y),tf.float32))
# ### Convert Pandas dataframe to matrix
df_train_feature_xport_m = df_train_feature_xport.values
df_train_label_xport_m   = df_train_label_xport.values
df_test_feature_xport_m = df_test_feature_xport.values
df_test_label_xport_m   = df_test_label_xport.values
train_num = len(df_train_label_xport_m)
test_num = len(df_test_label_xport_m)
# ### Set up constant hyperparameters for model
learning_rate=0.01
training_steps=1000
# ### Run my TensorFlow Graph and compute accuracy
graph = tf.Graph()
with graph.as_default():
    with tf.name_scope('Input'):
        x_train_ = tf.convert_to_tensor(df_train_feature_xport_m, dtype=tf.float32)
        y_train_ = tf.reshape(tf.convert_to_tensor(df_train_label_xport_m, dtype=tf.float32),[train_num,1])
        x_test_ = tf.convert_to_tensor(df_test_feature_xport_m, dtype=tf.float32)
        y_test_ = tf.reshape(tf.convert_to_tensor(df_test_label_xport_m, dtype=tf.float32),[test_num,1])
    with tf.name_scope('Variables'):
        # params and variables initialization
        # there are 21 features and 1 outcome
        W = tf.Variable(tf.zeros([21,1]), name='Weights') # 21 represents 21 features
        b = tf.Variable(0., name='Bias',)
    with tf.name_scope('Inference'):
        y = inference(x_train_)
    with tf.name_scope('Cross_Entropy'):
        cross_entropy = loss(x_train_,y_train_)
    with tf.name_scope('Train_Op') as scope:
        train_op = train(cross_entropy, learning_rate)
    with tf.name_scope('Evaluator') as scope:
        accuracy = evaluate(x_train_,y_train_)
        #correct_prediction is True/False boolean vector, cast converts to 1/0
    with tf.name_scope('Summaries'):
        summ_W = tf.summary.histogram('weights', W)
        summ_b = tf.summary.histogram('biases', b)
        summ_ce = tf.summary.scalar('cross_entropy', cross_entropy)
        summ_acc = tf.summary.scalar('accuracy', accuracy)
        summ_merged = tf.summary.merge([summ_W, summ_b, summ_ce, summ_acc])  
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter('./test11',sess.graph)
        coord=tf.train.Coordinator()
        threads=tf.train.start_queue_runners(sess=sess,coord=coord)
        for step in np.arange(training_steps):
            sess.run(train_op)
            #Finally, continuously print our progress and the final accuracy of the test data
            if step % 100 == 0:
                summary_str = sess.run(summ_merged)
                writer.add_summary(summary_str, step)
                print('Step =',step,' Accuracy =',sess.run(accuracy))
        print('Final Accuracy: ', sess.run(accuracy))
        print('done')
    coord.request_stop()
    coord.join(threads)
    writer.close()
    sess.close()
# ![TensorBoard](logistic.png)
# ![TensorBoard](logistic_summaries.png)
# ![TensorBoard](distributions.png)
# ![TensorBoard](histograms.png)
# ### Comment: Using full train/test and only the top 21 features with the highest feature importance from random forest model, this model in TensorFlow using sigmoid  function for cross entropy was able to reach a final accuracy of 93.58%, which is below the accuracy of the random forest model
# ## Running unsupervised K-Means on the full train/test dataset
# ### Create a new graph object
tf.reset_default_graph()
graph = tf.Graph()
# ### Set parameters for graph run
num_steps = 1000 # Total steps to train
k = 2 # The number of clusters
num_classes = 2 # 2 class outcomes
num_features = 21 # 9 features with highest feature importance from random forest
# ### Input features/labels(for assigning a label to a centroid and testing)
X_kmeans = tf.placeholder(tf.float32, shape=[None, num_features])
Y_kmeans = tf.placeholder(tf.float32, shape=[None, num_classes])
# ### Define K-Means
# K-Means Parameters
kmeans = KMeans(inputs=X_kmeans, num_clusters=k, distance_metric='cosine',
                use_mini_batch=True)
# Build KMeans graph
training_graph = kmeans.training_graph()
if len(training_graph) > 6: 
    (all_scores, cluster_idx, scores, cluster_centers_initialized,
     cluster_centers_var, init_op, train_op) = training_graph
else:
    (all_scores, cluster_idx, scores, cluster_centers_initialized,
     init_op, train_op) = training_graph
cluster_idx = cluster_idx[0] # fix for cluster_idx being a tuple
avg_distance = tf.reduce_mean(scores)
# ### Initialize variables and sessions
# Initialize the variables (i.e. assign their default value)
init_vars = tf.global_variables_initializer()
# Start TensorFlow session
sess = tf.Session()
# Run the initializer
sess.run(init_vars, feed_dict={X_kmeans: df_train_feature_xport_m})
sess.run(init_op, feed_dict={X_kmeans: df_train_feature_xport_m})
# ### Actual K-means training
for i in range(1, num_steps + 1):
    _, d, idx = sess.run([train_op, avg_distance, cluster_idx],
                         feed_dict={X_kmeans: df_train_feature_xport_m})
    if i % 100 == 0 or i == 1:
        print("Step %i, Avg Distance: %f" % (i, d))
# Assign a label to each centroid
# Count total number of labels per centroid, using the label of each training
# sample to their closest centroid (given by 'idx')
counts = np.zeros(shape=(k, num_classes))
for i in range(len(idx)):
    counts[idx[i]] += df_train_label_xport_m[i]
    
# Assign the most frequent label to the centroid
labels_map = [np.argmax(c) for c in counts]
labels_map = tf.convert_to_tensor(labels_map)
# ### K-means model evaluation
cluster_label = tf.nn.embedding_lookup(labels_map, cluster_idx) # Lookup: centroid_id -> label
# Compute accuracy
correct_prediction_kmeans = tf.equal(cluster_label, tf.cast(tf.argmax(Y_kmeans, 1), tf.int32))
accuracy_kmeans = tf.reduce_mean(tf.cast(correct_prediction_kmeans, tf.float32))
# Test Model
test_x_kmeans, test_y_kmeans = df_test_feature_xport_m ,pd.get_dummies(df_test_label_xport_m).values
print("K Means Test Accuracy:", sess.run(accuracy_kmeans, feed_dict={X_kmeans: test_x_kmeans, Y_kmeans: test_y_kmeans}))
# Open a SummaryWriter to save summaries
writer = tf.summary.FileWriter('./test_kmeans2', sess.graph)
# Write the summaries to disk
writer.flush()      
writer.close()
sess.close()
# ![TensorBoard](kmeans.png)
# ### Comment: Using full train/test and only the top 21 features with the highest feature importance from random forest model, K-means model in TensorFlow was able to reach a final accuracy of 90.88%, which is below the accuracy of both random forest model and TensorFlow logistic regression
# ## Final Conclusion
# ### For this Lending Club loan dataset, after all the data transformation and cleaning steps, we have attempted 3 different approches focusing on predicting which active borrowers are unlikely to be current on their loan obligations. Our findings are supervised learning algorithm outperforms unsupervised learning algorithm and random forest model implemented in Scikit Learn outperforms the logistic regression model implemented in TensorFlow.
