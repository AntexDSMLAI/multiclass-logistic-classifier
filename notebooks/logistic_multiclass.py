import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report

#Read Data
data = pd.read_csv('multiclass-logistic-classifier/data/raw/heart.csv')

#Perform SDA on the data to see how the data is 
print(data.head())
print(data.describe())
print(data.info())

print(data['target'].value_counts())

data.hist(bins = 20, figsize=(55,10))
plt.show()

#See if there is gigh corollation between the features

correlation_matrix = data.corr()

plt.figure(figsize=(15,10))
sns.heatmap(correlation_matrix,annot=True,cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

'''
Key Observations:

	1.	Positive Correlations:
	•	cp (chest pain type) shows a relatively strong positive correlation with the target (0.43). This suggests that certain types of chest pain are linked to heart disease (target = 1).
	•	thalach (maximum heart rate achieved) also has a positive correlation with the target (0.42), indicating higher heart rates might be associated with heart disease.
	2.	Negative Correlations:
	•	exang (exercise-induced angina), ca (number of major vessels), and oldpeak (ST depression induced by exercise) show negative correlations with the target, suggesting that heart disease tends to be less associated with these features (as their values increase, heart disease is less likely).
	•	slope has a significant positive relationship with the target, while thal (Thalassemia) has a negative correlation.
	3.	Feature Relationships:
	•	Strong positive correlations between thalach and slope, or negative correlations between oldpeak and thalach, suggest multicollinearity between some features. This could affect model performance.

What to Do Next:

	1.	Feature Selection:
	•	Based on the correlations, you might want to use cp, thalach, and slope as input features since they correlate well with the target.
	•	Consider dropping highly collinear features (such as thalach and oldpeak together) to reduce multicollinearity, which can improve model performance.
	2.	Scaling Features:
	•	Features like age, trestbps, and chol have different ranges. It would be good to standardize them using StandardScaler or MinMaxScaler from sklearn.
'''

X = data[['cp','thalach','slope','age','sex']]

y = data['target']


#split data into training and test with test size of 20%

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2 ,random_state=42)

print(X_train.shape)
print(X_test.shape)

#Let us now standardize the data 

scaler = StandardScaler()

X_train_scaled =scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(X_train_scaled)
print(X_test_scaled)




#Now train the logestic Regration with OVR

model = LogisticRegression(multi_class='ovr',solver='lbfgs',max_iter=500)

model.fit(X_train_scaled,y_train)


#Now Predict the target

y_predict = model.predict(X_test_scaled)


#evaluate the model

# Accuracy

print("The accuracy of the model is ",accuracy_score(y_test,y_predict))

#classfiction Report

print("Classification Report ",classification_report(y_test,y_predict))


#Confussion Matix

print("The confussion matrix of the model is ",confusion_matrix(y_test,y_predict))

plt.figure(figsize=(8,6))
sns.heatmap(confusion_matrix(y_test,y_predict), annot=True, cmap='Blues', fmt='d')
plt.title("Confusion Matrix")
plt.show()





