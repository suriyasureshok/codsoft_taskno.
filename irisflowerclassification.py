#Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

#Loading data
df = pd.read_csv("/kaggle/input/iris-flower-dataset/IRIS.csv",encoding='ISO-8859-1')
df

#Numerical Information
df.describe(include='all')

#Training The Model
X = df.drop(columns=['species'])  # Features
y = df['species']  

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

print("Training features shape:", X_train.shape)
print("Testing features shape:", X_test.shape)

log_reg_model=LogisticRegression(max_iter=200)
log_reg_model.fit(X_train,y_train)
y_pred_lr=log_reg_model.predict(X_test)
conf_matrix_lr=confusion_matrix(y_test,y_pred_lr)

rf_model=RandomForestClassifier(n_estimators=100,random_state=42)
rf_model.fit(X_train,y_train)
y_pred_rf = rf_model.predict(X_test)

print("Logistic Regression Model:")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lr))
print("Classification Report:\n", classification_report(y_test, y_pred_lr))

print("\nRandom Forest Model:")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
print("Classification Report:\n", classification_report(y_test, y_pred_rf))
accuracy_lr = accuracy_score(y_test, y_pred_lr)
print(f"Logistic Regression Model Accuracy: {accuracy_lr * 100:.2f}%")
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Model Accuracy: {accuracy_rf * 100:.2f}%")

#Heat Map
numeric_df = df.select_dtypes(include=[np.number])
correlation_matrix = numeric_df.corr()

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='Reds', fmt='.2f', cbar=True)
plt.title("Correlation Heatmap for Numeric Columns")
plt.show()

#Input Code(Commented) and Trial Data
#print("\nProvide the following details:")
#sepal_length = float(input("Enter the Sepal length: "))
#sepal_width = float(input("Enter the Sepal Width: "))
#petal_length = float(input("Enter the Petal Length: "))
#petal_width = float(input("Enter the Petal Width: "))

#data={
    #'sepal_length': [sepal_length],
    #'sepal_width' : [sepal_width],
    #'petal_length' : [petal_length],
    #'petal_width' : [petal_width],
#}
data={
    'sepal_length': [5],
    'sepal_width' : [3],
    'petal_length' : [1.5],
    'petal_width' : [0.2],
}

#Output
input_data = pd.DataFrame(data)
flower_type = rf_model.predict(input_data)[0]
print(f"\nFlower Classified as:", flower_type)
