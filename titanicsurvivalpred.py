#Importin Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.impute import SimpleImputer

#Loading The Data
df = pd.read_csv("/kaggle/input/titanic-dataset/Titanic-Dataset.csv",encoding='ISO-8859-1')
df

#Numerical Information
df.describe(include="all")

#Data Cleaning
df.drop(columns=['Name','Ticket','PassengerId','Fare'])
df['Sex'] = df['Sex'].replace({0: 'Male', 1: 'Female'})
df['Survived'] = df['Survived'].replace({1: 'Survived', 0: 'Not Survived'})
df.update

#Visualizataions of data
plt.figure(figsize=(12, 6))
sns.countplot(data=df, x='Sex', hue='Survived')
plt.title("Survival Rate by Gender")
plt.show()

plt.figure(figsize=(15, 5))
df[df['Survived'] == 'Survived']['Age'].hist(bins=30, edgecolor='black', alpha=0.7, label='Survived')
df[df['Survived'] == 'Not Survived']['Age'].hist(bins=30, edgecolor='black', alpha=0.7, label='Not Survived')

plt.suptitle('Age Distribution by Survival Status')
plt.xlabel('Age')
plt.ylabel('Number of Passengers')
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
sns.countplot(data=df, x='Pclass', hue='Survived')
plt.title("Survival Rate by Passenger Class")
plt.show()

#Training The Model
df_encoded = pd.get_dummies(df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']], drop_first=True)
X = df_encoded
y = df['Survived']

imputer = SimpleImputer(strategy='median')
X = imputer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

log_reg_model = LogisticRegression(max_iter=1000)
log_reg_model.fit(X_train, y_train)
y_pred_lr = log_reg_model.predict(X_test)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

print("Data types in the dataset:")
print(df.dtypes)
numeric_df = df.select_dtypes(include=[np.number])
numeric_df.fillna(numeric_df.median(), inplace=True)

correlation_matrix = numeric_df.corr()

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='Reds', fmt='.2f', cbar=True)
plt.title("Correlation Heatmap for Numeric Columns")
plt.show()

print("Logistic Regression Evaluation:")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lr))
print("Classification Report:\n", classification_report(y_test, y_pred_lr))

print("\nRandom Forest Evaluation:")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
print("Classification Report:\n", classification_report(y_test, y_pred_rf))

accuracy_lr = accuracy_score(y_test, y_pred_lr)
print(f"Logistic Regression Model Accuracy: {accuracy_lr * 100:.2f}%")
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Model Accuracy: {accuracy_rf * 100:.2f}%")

df = pd.DataFrame({
    'Pclass': [3, 1, 2],
    'Sex': ['male', 'female', 'male'],
    'Age': [22, 38, 26],
    'SibSp': [1, 1, 0],
    'Parch': [0, 0, 0],
    'Fare': [7.25, 71.2833, 8.05],
    'Embarked': ['S', 'C', 'S']
})
df_encoded = pd.get_dummies(df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']], drop_first=True)

X = df_encoded
y = [0, 1, 1] 
imputer = SimpleImputer(strategy='median')
X = imputer.fit_transform(X)

log_reg_model = LogisticRegression(max_iter=1000)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

log_reg_model.fit(X, y)
rf_model.fit(X, y)

#User Inputs
Pclass = int(input("Passenger Class (1, 2, 3): "))
Sex = input("Gender (Male, Female): ").strip().lower()
Sex = 1 if Sex == 'Male' else 0
Age = float(input("Age: "))
SibSp = int(input("Number of Siblings/Spouses Aboard: "))
Parch = int(input("Number of Parents/Children Aboard: "))
Fare = float(input("Fare: "))
Embarked = input("Port of Embarkation (Queenstown, Southampton): ").strip().upper()
Embarked_Q = 1 if Embarked == 'Queenstown' else 0
Embarked_S = 1 if Embarked == 'Southampton' else 0
Embarked_c = 1 if Embarked == 'Cherbourg' else 0

user_data = pd.DataFrame({
    'Pclass': [Pclass],
    'Sex_male': [Sex],
    'Age': [Age],
    'SibSp': [SibSp],
    'Parch': [Parch],
    'Fare': [Fare],
    'Embarked_Q': [Embarked_Q],
    'Embarked_S': [Embarked_S]
})

user_data = imputer.transform(user_data)

rf_pred = rf_model.predict(user_data)

print("\nPrediction Results:")
print(f"Random Forest Model Prediction: {'Survived' if rf_pred[0] == 1 else 'Not Survived'}")
