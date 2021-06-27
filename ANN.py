################### problem1 ###############################
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn import metrics

#load the dataset
diabetes = pd.read_csv("E:/ARTIFICIAL ASSIGNMENT/Deep Learning Chalenges/diabetes.csv")

#EDA
#checking for NA values and null values
diabetes.isna().sum()
diabetes.isnull().sum()

#identify duplicated records in the data
duplicate = diabetes.duplicated()
sum(duplicate)

#checking unique value for each columns
diabetes.nunique()

EDA  = {"column":diabetes.columns,
        "mean":diabetes.mean(),
        "median":diabetes.median(),
        "mode":diabetes.mode(),
        "standard deviation":diabetes.std(),
        "kurtosis":diabetes.kurt(),
        "skewness":diabetes.skew(),
        "variance":diabetes.var()}
EDA

#variance for each column
diabetes.var() 

#graphical representation
#histogram and scatter plot
sns.pairplot(diabetes, hue='Outcome')

#normalisation using z for all the continuous data
def norm_func(i):
    x = (i-i.mean()/i.std())
    return(x)

df = norm_func(diabetes.iloc[:,:8])

#final dataframe
final_diabetes = pd.concat([diabetes.iloc[:,[8]],df],axis = 1)

#train test splitting
np.random.seed(10)

final_diabetes_train,final_diabetes_test = train_test_split(final_diabetes, test_size = 0.2,random_state = 457) #20% test data

x_train = final_diabetes_train.iloc[:,1:].values.astype("float32")
y_train = final_diabetes_train.iloc[:,0].values.astype("float32")
x_test = final_diabetes_test.iloc[:,1:].values.astype("float32")
y_test = final_diabetes_test.iloc[:,0].values.astype("float32")

#model building
model = MLPRegressor(hidden_layer_sizes=(10,10,),activation='identity', max_iter=20 , solver = 'lbfgs')
model.fit(x_train,y_train)

#Evaluate the model on test data using mean absolute square error
mae1 = metrics.mean_absolute_error(y_test,model.predict(x_test))
print ("error on test data", mae1) 

# Evaluating the model on train data 
mae2 = metrics.mean_absolute_error(y_train, model.predict(x_train))
print("error on train data: ",mae2)

################################ problem2 ######################################################
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn import metrics
from sklearn import preprocessing
#load the dataset
churn_data = pd.read_csv("E:/ARTIFICIAL ASSIGNMENT/Deep Learning Chalenges/Churn_Modelling.csv")

#EDA
#checking for NA values and null values
churn_data.isna().sum()
churn_data.isnull().sum()

#identify duplicated records in the data
duplicate = churn_data.duplicated()
sum(duplicate)

#checking unique value for each columns
churn_data.nunique()

EDA  = {"column":churn_data.columns,
        "mean":churn_data.mean(),
        "median":churn_data.median(),
        "mode":churn_data.mode(),
        "standard deviation":churn_data.std(),
        "kurtosis":churn_data.kurt(),
        "skewness":churn_data.skew(),
        "variance":churn_data.var()}
EDA

#variance for each column
churn_data.var() 

#graphical representation
#histogram and scatter plot
sns.pairplot(churn_data, hue='Exited')

churn_data.columns

#Drop the unwanted columns
churn_data.drop(['RowNumber','CustomerId','Surname','Geography'],axis = 1, inplace = True)
 
#Label encoding
label_encoder = preprocessing.LabelEncoder()
churn_data['Gender'] = label_encoder.fit_transform(churn_data['Gender'])

#normalisation using z for all the continuous data
def norm_func(i):
    x = (i-i.mean()/i.std())
    return(x)

df = norm_func(churn_data.iloc[:,:8])

#final dataframe
final_churn_data = pd.concat([churn_data.iloc[:,[9]],df],axis = 1)

#train test splitting
np.random.seed(10)

final_churn_data_train,final_churn_data_test = train_test_split(final_churn_data, test_size = 0.2,random_state = 457) #20% test data

x_train = final_churn_data_train.iloc[:,1:].values.astype("float32")
y_train = final_churn_data_train.iloc[:,0].values.astype("float32")
x_test = final_churn_data_test.iloc[:,1:].values.astype("float32")
y_test = final_churn_data_test.iloc[:,0].values.astype("float32")

#model building
model = MLPRegressor(hidden_layer_sizes=(10,10,),activation='identity', max_iter=20 , solver = 'lbfgs')
model.fit(x_train,y_train)

#Evaluate the model on test data using mean absolute square error
mae1 = metrics.mean_absolute_error(y_test,model.predict(x_test))
print ("error on test data", mae1) 

# Evaluating the model on train data 
mae2 = metrics.mean_absolute_error(y_train, model.predict(x_train))
print("error on train data: ",mae2)

################################## problem3 ###########################################
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn import metrics
from sklearn import preprocessing
#load the dataset
breast_cancer = pd.read_csv("E:/ARTIFICIAL ASSIGNMENT/Deep Learning Chalenges/breast_cancer.csv")

#EDA
#checking for NA values and null values
breast_cancer.isna().sum()
breast_cancer.isnull().sum()

#identify duplicated records in the data
duplicate = breast_cancer.duplicated()
sum(duplicate)

#checking unique value for each columns
breast_cancer.nunique()

EDA  = {"column":breast_cancer.columns,
        "mean":breast_cancer.mean(),
        "median":breast_cancer.median(),
        "mode":breast_cancer.mode(),
        "standard deviation":breast_cancer.std(),
        "kurtosis":breast_cancer.kurt(),
        "skewness":breast_cancer.skew(),
        "variance":breast_cancer.var()}
EDA

#variance for each column
breast_cancer.var() 

breast_cancer.columns

#Drop the unwanted columns
breast_cancer.drop(['id','Unnamed: 32'], axis = 1,inplace = True)

#Perform label_encoding 
label_encoder = preprocessing.LabelEncoder()
breast_cancer['diagnosis'] = label_encoder.fit_transform(breast_cancer['diagnosis'])

#graphical representation
#histogram and scatter plot
sns.pairplot(breast_cancer, hue='diagnosis')

#normalisation using z for all the continuous data
def norm_func(i):
    x = (i-i.mean()/i.std())
    return(x)

df = norm_func(breast_cancer.iloc[:,1:])

#final dataframe
final_breast_cancer= pd.concat([breast_cancer.iloc[:,[0]],df],axis = 1)

#train test splitting
np.random.seed(10)

final_breast_cancer_train,final_breast_cancer_test = train_test_split(final_breast_cancer, test_size = 0.2,random_state = 457) #20% test data

x_train = final_breast_cancer_train.iloc[:,1:].values.astype("float32")
y_train = final_breast_cancer_train.iloc[:,0].values.astype("float32")
x_test = final_breast_cancer_test.iloc[:,1:].values.astype("float32")
y_test = final_breast_cancer_test.iloc[:,0].values.astype("float32")

#model building
model = MLPRegressor(hidden_layer_sizes=(10,10,),activation='tanh', max_iter=20 , solver = 'lbfgs')
model.fit(x_train,y_train)

#Evaluate the model on test data using mean absolute square error
mae1 = metrics.mean_absolute_error(y_test,model.predict(x_test))
print ("error on test data", mae1) 

# Evaluating the model on train data 
mae2 = metrics.mean_absolute_error(y_train, model.predict(x_train))
print("error on train data: ",mae2)

###########################################END ############################################

