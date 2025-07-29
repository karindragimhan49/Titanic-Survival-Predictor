# src/train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib
import os

# --- 1. Load Data ---
df = pd.read_csv('data/train.csv')

# --- 2. Clean Data ---
 
age_median = df['Age'].median()
df['Age'].fillna(age_median, inplace=True)
embarked_mode = df['Embarked'].mode()[0]
df['Embarked'].fillna(embarked_mode, inplace=True)
df.drop('Cabin', axis=1, inplace=True)
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

# --- 3. Feature Selection ---
X = df.drop(['Survived', 'PassengerId', 'Name', 'Ticket'], axis=1)
y = df['Survived']

# --- 4. Split and Train ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print(f"Model Accuracy: {model.score(X_test, y_test) * 100:.2f}%")

# --- 5. Save Model ---
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/titanic_model.joblib')

print("Model trained and saved to models/titanic_model.joblib")