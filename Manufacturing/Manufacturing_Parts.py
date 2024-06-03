#!/usr/bin/env python
# coding: utf-8

# ## Independent Variable 
# there are 5 independent variables: temperature, pressure, tempeture X pressure , material fusion, and material transformation. 
# 
# ## Dependent Variable
# Quality of the material in the dependent variable in this dataset.
# 
# 
# 
# 
# ## Background study on manufacturing terms
# 
# The quality of a material depends on many independent variable and in this report I am interested to see the specific effect of the temperature, pressure, temperature X pressure , material fusion, and material transormation. To do that I have done a bit of readings which I will share in this section which will the backbone of my report.
# 
# #### Thermal properties :
# There are 3 properties which corresponds to the concept of a material temperature. These properties include thermal expansion, thermal conductivity, and thermal stress. 
# 
# The temperature which a material can stand is called its thermal expansion property. Generally, when a material is heated, it gets expanded . This expansion can change the area,volume, and shape of the material. 
# The thermal conductivity of a material correspond its ability that it can conduct. 
# 
# The third important propery in relation to temperature is thermal stress. When heat is applied to a material it can undergo either expansion or contraction. 
# 
# Consider the below table that provide visualize thermal condutivity of different property:
# 
# ![image.png](attachment:image.png)
# 
# As you can see material conductivity changes at different temperature. Some materials have high thermal conductivity in low temperature and slightly lower temperature when heats go up such as Tungsten. 
# 
# 
# ### Pressure property:
# 
# When assessing material pressure property we can categorize them into metalic and non-metalic property. The key properties that provide basic understanding of a materual pressure propert include Strength, Elasticity, Plasticity, ULTIMATE TENSILE STRESS (UTS), ALLOWABLE (DESIGN) STRESS, DUCTILITY,TOUGHNESS, CHARPY IMPACT TESTING,HARDNESS. I will discuss about Strength, Elasticity, and Plasticity in this report.
# 
# #### 1. Strength: 
# The strength of a material is measured by pulling a test spicemen in tension until a fracture occurs . 
# ####  2. Elasticity: 
# "The ability of a deformed material body to return to its original shape and size when the forces causing the
# deformation are removed". It is important to mention that elasticity property has limitation. The below graph shows this property:
# 
# ![image-2.png](attachment:image-2.png)
# 
# #### 4.Plasticity:
# Once the material pass its elasticity limit, it enters its plasticity limit. As the material undergo its elongation within its plasticity range, it is now under plastic deformation. In this process, the material is getting stronger due to work hardening(strain hardening) which is a consequence of plastic deformation.  
# 
# 
# 
# 
# 
# 
# #### 4.Temperature x Pressure: 
# This feature is an interaction term between temperature and pressure, which captures the combined effect of these two process parameters.
# 
# #### 5. Material Fusion Metric:
# 
# Represents a material fusion-related measurement during the manufacturing process.
# 
# ##### Fusion = temperature^2 + pressure ^3
# 
# Many material fusion processes have optimal temperature and pressure ranges. Exceeding these ranges can actually degrade material properties. For example:
# 
#         Too High Temperature:
#         Might cause unwanted reactions, material degradation (such as thermal decomposition), or excessive internal stresses.
# 
#         Too High Pressure:
#         Could lead to structural weaknesses, like cracks or voids, especially if the material does not uniformly respond to pressure.
# 
# 
# 
# 
# 
# #### 6. Material Transformation Metric:
# 
# transformation = temperature^3 - pressure ^2
# 
# By cubing the temperature, this metric suggests a highly non-linear response of the material's transformation properties to changes in temperature. This could indicate scenarios where temperature has a more pronounced effect on the transformation than pressure.
# 
# The squared term for pressure, being subtracted, suggests that increasing pressure might counteract or reduce some of the effects induced by high temperatures. This could be relevant in processes where pressure is used to stabilize or control the material properties.
# 
# 
# 
# 
#  It provides insight into material transformation dynamics.
# 
# Resources:
# https://byjus.com/physics/thermal-properties-of-materials/#:~:text=a%20railway%20track-,Thermal%20conductivity,the%20ones%20with%20low%20conductivity.
# 
# https://wilkinsoncoutts.com/back-to-basics-material-properties/
# 

# # EDA

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler,PolynomialFeatures

dataframe = pd.read_csv("https://raw.githubusercontent.com/mpaydar/Regression_Modeling/main/Manufacturing/manufacturing.csv")
dataframe.head()


# ## Elementary Inspection of the Dataset:
# 
# ### 1. Number of Rows and columns 
# ### 2. DataTypes
# ### 3. Statistical information about the attributes

# In[2]:


rows,columns=dataframe.shape
print(f'Number of  rows: {rows} \nNumber of columns {columns}')


# We are working with five independent variables and 1 dependent variable as previously mentioned.

# In[3]:


dataframe.info()


# Looks like we are only working with float data type. 

# In[4]:


dataframe.describe()


# After inspecting the initial inspection of values for each attributes, it looks like we have big gaps between all the attributes which require some regularization as part of the preprocessing phase. 

# In[5]:


column_names=dataframe.columns
column_names


# ## Regularizing the dataframe using MinMax()

# In[6]:


scaler=MinMaxScaler()
regularized_dataframe=scaler.fit_transform(dataframe)
dataframe=pd.DataFrame(regularized_dataframe,columns=column_names)
dataframe


# ### Boxplots of Independent Variables

# In[7]:


plt.figure(figsize=(15,10))
independent_variables=dataframe.drop(columns='Quality Rating')

sns.boxplot(data=independent_variables)
plt.show()


# looks like there are few tuples that are outliers in temperature X pressure. The best approach to handle these few outliers is the dropping of these rows since it is not a significant number. 

# In[8]:


# Calculating Q1, Q3, and IQR
Q1 = dataframe['Temperature x Pressure'].quantile(0.25)
Q3 = dataframe['Temperature x Pressure'].quantile(0.75)
IQR = Q3 - Q1

# Defining outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR


dataframe = dataframe[(dataframe['Temperature x Pressure'] > lower_bound) & (dataframe['Temperature x Pressure'] < upper_bound)]
dataframe





# After filtering the tuples for the outliers, we have lost 6 of the tuples from the dataset which does not result in significant loss of information as it is important when choosing the strategy to handle the outliers. 
# 
# Now let's see a correlation tables which can help us to select the most relavant features to quality rating. Selecting the most correlated attributes helps us to develop a more optimize data model. 

# ## Correlation Matrix

# In[9]:


dataframe.corr()


# Upon inspection of correlation matrix, the strongest correlation is material transformation followed by  material fusion, temperature , and temperature X Pressure. Although these are among the stronger correlation, they are negative. For the purpose of the polynomial model, we want to select **Material   Transformation Metric, Material Fusion Metric, and Temperature (°C)**. 
# 
# 
# ## Feature Selection

# In[10]:


selected_featurs=dataframe[['Material Transformation Metric','Material Fusion Metric','Temperature (°C)']]
selected_featurs


# In[11]:


# Set the size of the overall figure
plt.figure(figsize=(15, 5))  # Adjust size to your preference

# Loop through each selected feature
for i, feature in enumerate(selected_featurs.columns):
    plt.subplot(1, len(selected_featurs.columns), i + 1)  # Creates a subplot for each feature
    plt.scatter(selected_featurs[feature], dataframe['Quality Rating'], alpha=0.5)  # Plot scatter
    plt.title(f'Scatter Plot of {feature} vs Quality Rating')  # Set title
    plt.xlabel(feature)  # Label x-axis
    plt.ylabel('Quality Rating')  # Label y-axis

plt.tight_layout()  # Adjust subplots to fit into figure area.
plt.show()  # Display the plots


# In[16]:


# plt.scatter(selected_featurs['Temperature (°C)'], dataframe['Quality Rating'], alpha=0.5)  # Plot scatter
poly=PolynomialFeatures(3)
selected_feature_columns= selected_featurs.columns
X_Poly=poly.fit_transform(selected_featurs)
X_Poly.shape


# In[22]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

X_train, X_test, y_train, y_test = train_test_split(X_Poly, dataframe['Quality Rating'], test_size=0.3, random_state=42)
model_one=LinearRegression()
r=model_one.fit(X_train,y_train)
print(f'Optimized-Slope:{r.coef_}  \nOptimized-Y-Intercept: {r.intercept_}')


# In[28]:


y_predictions=model_one.predict(X_test)

plt.scatter(X_test[:,1],y_test)
plt.scatter(X_test[:,1],y_predictions)
plt.show()


# In[29]:


from sklearn.metrics import mean_squared_error


mse = mean_squared_error(y_test, y_predictions)
mse


# In[ ]:




