#!/usr/bin/env python
# coding: utf-8

# 
# # Project: Investigate a Dataset - No show appointments
# 
# ## Table of Contents
# <ul>
# <li><a href="#intro">Introduction</a></li>
# <li><a href="#wrangling">Data Wrangling</a></li>
# <li><a href="#eda">Exploratory Data Analysis</a></li>
# <li><a href="#conclusions">Conclusions</a></li>
# </ul>

# <a id='intro'></a>
# ## Introduction
# 
# ### Dataset Description 
# This dataset collects information from 100k medical appointments in Brazil and is focused on the question of whether or not patients show up for their appointment. A number of characteristics about the patient are included in each row. The dataset has fourteen columns as  follows:
# 
# 1. PatientId - indicates the Patient identification no. It is unique to each patient. 
# 2. AppointmentID - tells us the identifiable number for each appointment of the patients. 
# 3. Gender  -  specifies whether a patient is a male or female. 
# 4. ScheduledDay  - tells us on what day the patient set up their appointment. 
# 5. AppointmentDay - Indicates the day of appointment. 
# 6. Age -  tells us about the  patients' age  which indicates whether the patient is an infact, young or adult. 
# 7. Neighbourhood - indicates the location of the hospital. 
# 8. Scholarship -indicates whether or not the patient is enrolled in Brasilian welfare program Bolsa Família. 
# 9. Hipertension - indicates whether the patient is hypertensive or not. 
# 10. Diabetes - indicates whether the patient is diabetic or not. 
# 11. Alcoholism -  tells us whether the patient takes alcohol or not. 
# 12. Handcap - indicates the physical disability of the patient. 
# 13. SMS_received -  tells us whether a patient receives SMS notification or not. 
# 14. No-show -: it says ‘No’ if the patient showed up to their appointment, and ‘Yes’ if they did not show up. 
# 
# 
# ### Question(s) for Analysis
# Under this analysis, the general question is What factors are important for us to know in order to predict if a patient will show up for their scheduled appointment? But in specific term, this analysis will try to analysis the following questions.
# 1. Is there any association between patient's gender and show up for their scheduled appointment?
# 2. Does age of the patient affect their showing up for the scheduled appointment?
# 3. Is scholarship a factor to determine if a patient will show up for their scheduled appointment?
# 

# In[ ]:


# import statements for all of the packages relevant for the analysis
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import datetime


# <a id='wrangling'></a>
# ## Data Wrangling
# 
# 
# 
# ### General Properties

# #### Load Data
# > The data is loaded from the CSV file in to a DataFrame and then check few rows to examine the data.

# In[ ]:


df = pd.read_csv('noshowappointments-kagglev2-may-2016.csv')
df.head()


# In[ ]:


# Print the last past of the data
df.tail()


# In[ ]:


# Print summary of the basic description of the data
df.describe()


# >From the above, the average age of the patient is 37.08, the minimum age is negative 1 ( which is abnormal) and maximum age is 115 years.  Let us see other features of each column

# In[ ]:


# Print basic information about the data
df.info()


# >Further review of the data shows that the dataframe contains 14 columns and 110527 rows. There are no missing data as shown above.

# In[ ]:


# Examine the unique values in some of the columns:

print("Unique values in Gender column are {}".format(df.Gender.unique()))
print("Unique values in Scholarship column are {}".format(df.Scholarship.unique()))
print("Unique values in Hipertension column are {}".format(df.Hipertension.unique()))
print("Unique values in Diabetes column are {}".format(df.Diabetes.unique()))
print("Unique values in Alcoholism column are {}".format(df.Alcoholism.unique()))
print("Unique values in Handcap column are {}".format(df.Handcap.unique()))
print("Unique values in SMS_received column are {}".format(df.SMS_received.unique()))


# 
# ### Data Cleaning
# 

# >Data Features and Issues:
# >The following are the observed features in the dataset  that require corrections:
#     1. PatientId is an Integer and not Float. So, we will convert it into int64.
#     2. Data Type of ScheduledDay and AppointmentDay will be changed to DateTime.
#     3. Spelling error and typo in the Column names. "No-show" and "Sms_Received" columns will be renamed as lines of codes            where they are appear might run into error. Others do not post serious challenge  but they will still be renamed.
#     4. As the AppointmentDay has 00:00:00 in it's TimeStamp, we will ignore it as it does not affect our analysis.
#     5. The following columns that contain YES or NO kind of values will be changed to object type:
#         Scholarship, 
#         Hipertension,
#         Diabetes,         
#         Alcoholism ,     
#         Handcap, 
#         SMS_received,
#     6. Age contains negative figure which is abnormal . The row that contains negative number will be deleted.
#     7. PatientId and AppointmentID  columns are not relevant for further analysis, hence they will be dropped.

# In[ ]:


# check if the data contains duplicate
sum(df.duplicated())


# > the result of duplicate check is zero, it follows that the data does not contain duplicate.

# In[ ]:


# check for data information 
df.info()


# In[ ]:


# Convert PatientId from Float to Integer
df['PatientId'] = df['AppointmentID'].astype('int64')


# In[ ]:


# Convert ScheduledDay and AppointmentDay from 'object' type to 'datetime64[ns]'
df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay']).dt.date.astype('datetime64[ns]')
df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay']).dt.date.astype('datetime64[ns]')


# In[ ]:


# Renaming of incorrect column names:
df = df.rename(columns={'Hipertension': 'Hypertension', 'Handcap': 'Handicap', 
                        'SMS_received': 'SMSReceived', 'No-show': 'NoShow'})

# Change all the columns name to lower case for easy reference
df= df.rename(columns=str.lower)


# In[ ]:


#  COnfirm if the above modification achieve intended results on the data set
df.info()


# >The spelling errors in the columns names have been addressed as shown above and also scheduledDay and AppointmentDay have been changed to datetime type. Next is to change scholarship, hypertension, diabetes, handicap, alcoholism, smsreceived to object as they contain YES or No kind of values

# In[ ]:


# Converting some columns  from int64 to object

df['scholarship'] = df['scholarship'].astype('object')
df['hypertension'] = df['hypertension'].astype('object')
df['diabetes'] = df['diabetes'].astype('object')
df['alcoholism'] = df['alcoholism'].astype('object')
df['handicap'] = df['handicap'].astype('object')
df['smsreceived'] = df['smsreceived'].astype('object')


# In[ ]:


#  COnfirm the result 
df.info()


# In[ ]:


# Obtain the unique values in column Age sorted in ascending order
np.sort(df.age.unique())


# >The result shows that column 'Age' contains -1 (Negative one). This is abnormal as patient age cannot be negative. Further investigate is required to confirm  the number of rows this affect. The patients will zero as their age value can be assumed to be infants who are less that one year in age, this shall be further reviewed. While the patient with age value of 115 appear unusual, though there are peole who live to that age but it is rare.

# In[ ]:


# Sumamry of Data based on Age Count.
df.groupby('age')['age'].value_counts()


# >From the result above patient with -1 as age is 1 which is insignificant  compare to the data size, hence this will be deleted.
# Patients  who are zero year of age is 3539 which is significant. These are babies who are few months old. While those with age above 115 are 5. 

# In[ ]:


# Drop the row that contains -1 as Age
df = df[df.age != -1]
df.info()


# In[ ]:


# Obtain the  Unique Values for 'Neighbourhood'
print("Neighbourhood are as follows: {}".format(np.sort(df.neighbourhood.unique())))


# In[ ]:


# Print count of the unique value of 'Neighbourhood'
df.neighbourhood.nunique()


# In[ ]:


# Drop 'PatientId' and 'AppointmentID' as they are not relevant for analysis since no duplicate
df.drop(['patientid', 'appointmentid'], axis=1, inplace=True)
df.head()


# In[ ]:


# Defining the function to get statistics  from Input data

def get_statistics(data, bins=20):
    total = data.values
    print('Mean:', np.mean(total))
    print('Standard deviation:', np.std(total))
    print('Minimum:', np.min(total))
    print('Maximum:', np.max(total))
    print('Median:', np.median(total))
    plt.hist(data, bins=bins);


# <a id='eda'></a>
# ## Exploratory Data Analysis
# 
# 
# > In this dataset, 'Noshow'column is the dependent variable, while the following are independent variables. 
# Gender           
# ScheduledDay      
# AppointmentDay    
# Age              
# Neighbourhood    
# Scholarship       
# Hypertension 
# Diabetes        
# Alcoholism  
# Handicap 
# SMSReceived 
# We will  examine how they affect whether patients will show up for their scheduled appointment  or not 
# 
# ### Research Question 1 : Patient's Gender  and Scheduled Appointment

# In[ ]:


# Print the unique value of 'Gender'
df.gender.unique()


# In[ ]:


# Print Gender Distribution of Patients

df['gender'].value_counts().plot(kind='pie', figsize= (8,8));
plt.title("Gender Distribution of Patients")
plt.show()


# In[43]:


# Sumamry of Data based on Gender Count.
df.groupby('gender')['noshow'].value_counts().plot(kind='bar')
plt.title("Gender Analysis of Show/NoShow for Appointment")
plt.xlabel("Gender")
plt.ylabel("Count")

plt.show()


# >**Gender and Scheduled Appointment** :From the above chart,'Female' patients  are more than 'Male' patients both in appointment and responses. So, Gender might be an important factor. But a closer look at the noShow distribution across Male's and Female's it is almost the same.  It follows that only gender might not be a determining faction to predict if a patient will show up for scheduled appointment or not. Probably if gender is combined with other factors , it might become significant in predicting whether a patient will honour his/her scheduled appointment or not.

# ### Research Question 2  : Patient's Age and Showing Up for Scheduled Appointment

# In[44]:


# Split the dataset into Show and Noshow for Appointment
show = df.noshow == 'No'
noshow = df.noshow == 'Yes'


# In[45]:


# Compute the basic statistics about the field "Age"
get_statistics(df.age)


# In[46]:


# Print the mean age of patients 

print("Mean Age of Patients who show for Appointment =>  {}".format(df.age[show].mean()))
print("Mean Age of Patients who Does not show for Appointment =>  {}".format(df.age[noshow].mean()))


# In[47]:


#  Analysis of patients who show or no show for appointment  by Age

df.age[show].hist(alpha=0.7, width = 2,figsize=(16,7),bins=80, label='Show')
df.age[noshow].hist(alpha=0.7, width = 2,figsize=(16,7),bins=80, label='NoShow')
plt.legend()
plt.title("Age Analysis of Show/NoShow for Appointment")
plt.xlabel("Age")
plt.ylabel("Count")

plt.show()


# >**Age Analysis**:
#     >The chart above depicts the age analysis of the patients vis-a-vis their scheduled appointment. From the analysis above,   
#     we can see that mean age of patients who show up  for their appointment is 37.7 while those that did not show up for their 
#     appointment is 34.31.  Moreso, patients whose age is zero(o) they show up for their appointment more than other age groups. 
#     In the same vien, in analysing the noshow , the age group zero(0) has the highest frequency of no show for appointment. 
#     Given that the mean for both no show and show falls within the same range and exhibits the same trend on both analysis, it 
#     follows that age of the patient is associated with if a patient will show up for scheduled appointment or 
#     not.

# ### Research Question 3  : Scholarship and Showing Up for Scheduled Appointment
# >Under this category, we analyse whether sholarship is associated  with scheduled appointment or not. 

# In[48]:


# Obtain the  Unique Values for 'Scholarship'
df.scholarship.unique()


# In[49]:


# Graph to show relationship between Scholarship and Scheduled Appointment (Noshow)
ax = sns.countplot(x=df.scholarship, hue=df.noshow, data=df)
ax.set_title("Show/NoShow for Scholarship")
x_ticks_labels=['No Scholarship', 'Scholarship']
ax.set_xticklabels(x_ticks_labels)
plt.show()


# In[50]:


# Gender and Scholarship with  No show

df.groupby('gender')['scholarship'].value_counts().plot(kind= 'bar', title= "Analysis of Gender and Scholarshp")


# In[51]:


df.groupby('scholarship')['noshow'].value_counts()


# **Scholarship and Noshow **: From the Analysis above , it could be seen that the patients with scholarship attended their  scheduled appointment. When further analysed based on gender, female with scholarship  show up for their scheduled appointment than their male counterpart. Also, patients without scholarship also show up for their scheduled appointment. So, Scholarship feature could help us in determining if a patient will show up for the visit after an appointment.

# <a id='conclusions'></a>
# ## Conclusions
# 
# ### Result
# > In the analysis we try to provide answers to the following research questions:
# 1. Is there any association between patient's gender and show up for their scheduled appointment?
# 2. Does age of the patient affect their showing up for the scheduled appointment?
# 3. Is scholarship  a factor to determine if a patient will show up for their scheduled appointment?
# 
# We examine, the relationship among the variables and found out that:
# 1. There exist association between patient's gender and noshow.
# 2. Analysing patient's age and their response to scheduled appointment, we observe that age is a factor in determining whether a patient will show or not show for an apopintment. As patients of certain age category responded positively to appointment than other.
# 3. Scholarship feature could help us in determining if a patient will show up for the visit after an appointment.
# 4. However, further analysing is required to establish further causal relationship among the variables. 
# . 
# 
# ### Limitations
# 1. AppointmentDay spans over a month  which means that the data is not representive and  predictions made based on this sample 
#     might be spurious. It might not be representation of the whole data.
#     
# 2. The reason for the appointment  and specialization of the doctors were not stated. These could have helped a lot in making better analysis and prediction for no show of a patient.

# In[52]:


from subprocess import call
call(['python', '-m', 'nbconvert', 'Investigate_a_Dataset.ipynb'])


# In[ ]:




