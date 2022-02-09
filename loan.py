#!/usr/bin/env python
# coding: utf-8

# In[107]:


import numpy as np
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt 

get_ipython().run_line_magic('matplotlib', 'inline')


# ##### read csv file

# In[3]:


loan=pd.read_csv('E:/upgrad/data/caseStudy/loan.csv',low_memory=False)
print(loan.head())


# In[4]:


#ptint list of columns 
loan.info(verbose = True)


# ##### data cleaning

# In[5]:


# Function_name : missingdata_percentage
# Usage : Returns % of missing values for all features in a DataFrame
# Arguments : dataframe
# Returns : a dataframe containing categories having missing values and % of missing values in those categories

def missingdata_percentage(df):
    missing = pd.DataFrame(columns=['category','percentage'])
    for col in df.columns:
        if df[col].isna().values.any():
            percentage = 100*df[col].isna().sum()/df.shape[0]
            missing = missing.append({'category' : col, 'percentage' : percentage}, ignore_index=True)
    return missing

missingData=missingdata_percentage(loan)

missingData


# In[6]:


pd.set_option('display.max_rows', None)
missingData.sort_values('percentage',ascending=False)


# ##### drop columns which has 100% no data in columns

# In[ ]:


# (bc_open_to_buy,acc_open_past_24mths,il_util,total_cu_tl,inq_fi,total_rev_hi_lim,all_util,max_bal_bc,open_rv_24m,
# open_rv_12m,total_bal_il,tot_cur_bal,open_il_24m,open_il_12m,open_il_6m,open_acc_6m,tot_coll_amt,verification_status_joint,
# dti_joint,annual_inc_joint,mths_since_last_major_derog,mths_since_rcnt_il,inq_last_12m,total_il_high_credit_limit,avg_cur_bal
# total_bc_limit,total_bal_ex_mort,tot_hi_cred_lim,percent_bc_gt_75,pct_tl_nvr_dlq,num_tl_op_past_12m,num_tl_90g_dpd_24m,
# num_tl_30dpd,num_tl_120dpd_2m,num_sats,num_rev_tl_bal_gt_0,num_rev_accts,num_op_rev_tl,num_il_tl,num_bc_tl,num_bc_sats,
# num_actv_rev_tl,num_actv_bc_tl,num_accts_ever_120_pd,mths_since_recent_revol_delinq,mths_since_recent_inq,mths_since_recent_bc_dlq,
# mths_since_recent_bc,mort_acc,mo_sin_rcnt_tl,mo_sin_rcnt_rev_tl_op,mo_sin_old_rev_tl_op,mo_sin_old_il_acct,bc_util,
# acc_open_past_24mths,bc_open_to_buy)
# (next_pymnt_d,mths_since_last_record) 97%
# mths_since_last_delinq 67%


# In[62]:


loan.drop(['bc_open_to_buy','acc_open_past_24mths','il_util','total_cu_tl','inq_fi','total_rev_hi_lim','all_util','max_bal_bc','open_rv_24m',
'open_rv_12m','total_bal_il','tot_cur_bal','open_il_24m','open_il_12m','open_il_6m','open_acc_6m','tot_coll_amt','verification_status_joint',
'dti_joint','annual_inc_joint','mths_since_last_major_derog','mths_since_rcnt_il','inq_last_12m','total_il_high_credit_limit','avg_cur_bal'
,'total_bc_limit','total_bal_ex_mort','tot_hi_cred_lim','percent_bc_gt_75','pct_tl_nvr_dlq','num_tl_op_past_12m','num_tl_90g_dpd_24m',
'num_tl_30dpd','num_tl_120dpd_2m','num_sats','num_rev_tl_bal_gt_0','num_rev_accts','num_op_rev_tl','num_il_tl','num_bc_tl','num_bc_sats',
'num_actv_rev_tl','num_actv_bc_tl','num_accts_ever_120_pd','mths_since_recent_revol_delinq','mths_since_recent_inq','mths_since_recent_bc_dlq'
,'mths_since_recent_bc','mort_acc','mo_sin_rcnt_tl','mo_sin_rcnt_rev_tl_op','mo_sin_old_rev_tl_op','mo_sin_old_il_acct','bc_util',
'acc_open_past_24mths','bc_open_to_buy','next_pymnt_d','mths_since_last_record','mths_since_last_delinq'],axis=1,inplace=True)


# In[64]:


loan.dropna(subset=['emp_length','emp_title','pub_rec_bankruptcies','last_pymnt_d','collections_12_mths_ex_med'],axis=0, inplace=True)


# In[9]:



loan.info()


# In[12]:


#find the categorical columns inside the dataframe and storede in columns
categorical_col=[cname for cname in loan.columns if loan[cname].dtype=="O"]
categorical_col


# In[10]:


#find the columns  missing values

colsWithMissingValues=[col for col in loan.columns if loan[col].isnull().sum()>1]
colsWithMissingValues


# In[52]:


for col in colsWithMissingValues:
    loan["col"]=np.where(loan[col].isnull(),1,0) 


# In[53]:


import seaborn as sns 
sns.countplot(loan["loan_status"])


# In[ ]:


#In the above we can see the data .if we use data out of 10 peredicition we will get approx 70% of time preson will not 
# default the bank.


# In[54]:


loan.loan_status.value_counts()


# In[56]:


loan.isnull().sum()


# In[68]:


#correlation with columns
loan.corr()


# In[84]:


#
loan.groupby([loan['loan_amnt'][:500],"loan_status"]).size().unstack().plot(kind='bar',stacked=True,figsize=(30,20))


# In[ ]:


# in the above graph the applications with higher loan amount have great chance of loan default 


# In[85]:


#loan default by instrest rate

loan.groupby([loan['int_rate'][:500],"loan_status"]).size().unstack().plot(kind='bar',stacked=True,figsize=(30,20))


# In[ ]:


#there is apositive correlation between the high instrest rate and loan default as intrest rate increase the chance of default 


# In[86]:


loan[['home_ownership','loan_status']].corr()


# In[87]:


loan[['annual_inc','loan_status']].corr()


# In[114]:


#total clamin amount distribution
import plotly.offline as py 
py.init_notebook_mode(connected=True) # this code, allow us to work with offline plotly version
import plotly.graph_objs as go # it's like "plt" of matplot
import plotly.tools as tls # It's useful to we get some tools of plotly
import warnings # This library will be used to ignore some warnings
from collections import Counter # To do counter of some features



fig=py.histogram(loan, x="funded_amnt_inv",columns='loan_status',marginal="box",hover_data=loan.columns)
fig.show


# In[ ]:


#finding the  grater values from all columns


# In[ ]:





# In[116]:


sns.set_style('dark')
loan.hist(bins=50,figsize=(20,20),color='navy')

