#Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import confusion_matrix, r2_score,mean_squared_error,mean_absolute_error
from sklearn.impute import SimpleImputer

#Loading data
df = pd.read_csv("/kaggle/input/imdb-india-movies/IMDb Movies India.csv", encoding='ISO-8859-1')
df

#Numerical INformations
df.info()
print(df.shape)
df.describe(include='all')

#Data Cleaning
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)
df['Year'] = df['Year'].astype(str)
df['Year'] = df['Year'].str.extract('(\d+)').astype(int)
df['Duration'] = df['Duration'].astype(str)
df['Duration'] = df['Duration'].str.extract('(\d+)').astype(int)

#averaging the column elements
df['Genre_Average_Rating'] = df.groupby('Genre')['Rating'].transform('mean')
df['Director_Average_Rating'] = df.groupby('Director')['Rating'].transform('mean')
df['Actor1_Average_Rating'] = df.groupby('Actor 1')['Rating'].transform('mean')
df['Actor2_Average_Rating'] = df.groupby('Actor 2')['Rating'].transform('mean')
df['Actor3_Average_Rating'] = df.groupby('Actor 3')['Rating'].transform('mean')


X = df[['Year', 'Votes', 'Duration', 'Genre_Average_Rating', 'Director_Average_Rating', 'Actor1_Average_Rating', 'Actor2_Average_Rating', 'Actor3_Average_Rating']]
y = df['Rating']
print("Unique values in 'Rating' column before cleaning:", df['Rating'].unique())
print("Unique values in 'Rating' column after cleaning:", df['Rating'].unique())
print("Unique values in 'Votes' column before cleaning:", df['Votes'].unique())
df['Votes'] = pd.to_numeric(df['Votes'], errors='coerce')
print("Unique values in 'Votes' column after cleaning:", df['Votes'].unique())
print("Unique values in 'Year' column before cleaning:", df['Year'].unique())
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
print("Unique values in 'Year' column after cleaning:", df['Year'].unique())
print("Unique values in 'Duration' column before cleaning:", df['Duration'].unique())
df['Duration'] = pd.to_numeric(df['Duration'], errors='coerce')
print("Unique values in 'Duration' column after cleaning:", df['Duration'].unique())

categorical_columns = []
df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
df = df.select_dtypes(include=[np.number])
df = df.dropna(subset=['Rating'])

X = df.drop(columns=['Rating'])
y = df['Rating']

imputer = SimpleImputer(strategy='median')
X = imputer.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

random_forest_model = RandomForestRegressor(random_state=42)
linear_regression_model = LinearRegression()

random_forest_model.fit(X_train, y_train)
linear_regression_model.fit(X_train, y_train)

y_pred_rf = random_forest_model.predict(X_test)
y_pred_lr = linear_regression_model.predict(X_test)

mse_rf = mean_squared_error(y_test, y_pred_rf)
mse_lr = mean_squared_error(y_test, y_pred_lr)

print("Data types in the dataset:")
print(df.dtypes)
numeric_df = df.select_dtypes(include=[np.number])
numeric_df.fillna(numeric_df.median(), inplace=True)

correlation_matrix = numeric_df.corr()

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='Reds', fmt='.2f', cbar=True)
plt.title("Correlation Heatmap for Numeric Columns")
plt.show()

print(f"Random Forest Mean Squared Error: {mse_rf}")
print(f"Linear Regression Mean Squared Error: {mse_lr}")

#Visualizations
plt.style.use('seaborn-v0_8-white')
df[['Year', 'Duration', 'Votes']].hist(bins=30, edgecolor='black', figsize=(10, 5))
plt.suptitle('Histograms of Numeric Features')
plt.show()

df['Rating'].hist(bins=30, edgecolor='black', figsize=(10, 5))
plt.suptitle('Distribution of Rating')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.show()

df = pd.read_csv("/kaggle/input/imdb-india-movies/IMDb Movies India.csv", encoding='ISO-8859-1')
top_10_directors = df['Director'].value_counts().head(10)
plt.figure(figsize=(10, 5))
sns.barplot(x=top_10_directors.values, y=top_10_directors.index, palette='Dark2')
plt.title('Top 10 Directors with Most Movie Involvements')
plt.xlabel('Number of Movies')
plt.ylabel('Director')
plt.show()

top_10_genres = df['Genre'].value_counts().head(10)
plt.figure(figsize=(10, 5))
sns.barplot(x=top_10_genres.values, y=top_10_genres.index, palette='muted')
plt.title('Top 10 Movie Genres')
plt.xlabel('Number of Movies')
plt.ylabel('Genre')
plt.show()

plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_pred_lr, alpha=0.5)
plt.title('Linear Regression Model: Actual vs Predicted Ratings')
plt.xlabel('Actual Ratings')
plt.ylabel('Predicted Ratings')
plt.show()

plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_pred_rf, alpha=0.5)
plt.title('Random Forest Model: Actual vs Predicted Ratings')
plt.xlabel('Actual Ratings')
plt.ylabel('Predicted Ratings')
plt.show()

y_pred_rf = random_forest_model.predict(X_test)
print("Performance Evaluation for Random Forest Model:")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred_rf)}")
print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred_rf)}")
print(f"R2 Score: {r2_score(y_test, y_pred_rf)}")

#input code(commented) and trial data 
#print("\nProvide the following details to predict the movie rating:")
#year = int(input("Enter the Year: "))
#votes = int(input("Enter the number of Votes: "))
#duration = int(input("Enter the Duration (in minutes): "))
#genre = input("Enter the Genre: ")
#director = input("Enter the Director: ")
#actor1 = input("Enter the Actor 1: ")
#actor2 = input("Enter Actor 2: ")
#actor3 = input("Enter Actor 3: ")

year = 1998
votes = 2444545
duration = 134
genre = 'Drama','Romance'
director ='Mahesh Bhatt'
actor1='Amitabh Bhachan'
actor2 = 'Rekha'
actor3 = 'nil'

#Analysis and Output
genre_avg_rating = df.groupby('Genre')['Rating'].mean().to_dict()
director_avg_rating = df.groupby('Director')['Rating'].mean().to_dict()
actor1_avg_rating = df.groupby('Actor 1')['Rating'].mean().to_dict()
actor2_avg_rating = df.groupby('Actor 2')['Rating'].mean().to_dict()
actor3_avg_rating = df.groupby('Actor 3')['Rating'].mean().to_dict()

genre_avg_rating_value = genre_avg_rating.get(genre, df['Rating'].mean())
director_avg_rating_value = director_avg_rating.get(director, df['Rating'].mean())
actor1_avg_rating_value = actor1_avg_rating.get(actor1, df['Rating'].mean())
actor2_avg_rating_value = actor2_avg_rating.get(actor2, df['Rating'].mean())
actor3_avg_rating_value = actor3_avg_rating.get(actor3, df['Rating'].mean())

print("Average rating for genre:", genre_avg_rating_value)
print("Average rating for director:", director_avg_rating_value)
print("Average rating for actor 1:", actor1_avg_rating_value)
print("Average rating for actor 2:", actor2_avg_rating_value)
print("Average rating for actor 3:", actor3_avg_rating_value)

data = {
    'Year': [year],
    'Votes': [votes],
    'Duration': [duration],
    'Genre_Average_Rating': [genre_avg_rating_value],
    'Director_Average_Rating': [director_avg_rating_value],
    'Actor1_Average_Rating': [actor1_avg_rating_value],
    'Actor2_Average_Rating': [actor2_avg_rating_value],
    'Actor3_Average_Rating': [actor3_avg_rating_value]
}

input_data = pd.DataFrame(data)
predicted_rating = random_forest_model.predict(input_data)[0]
print(f"\nPredicted IMDB:", predicted_rating)
