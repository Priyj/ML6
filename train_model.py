
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import pickle

# Load dataset
df = pd.read_csv("synthetic_social_addiction.csv")

# Clean and engineer features
df = df.dropna(subset=[
    'Platform', 'Total Time Spent', 'Number of Sessions',
    'Engagement', 'Scroll Rate', 'Addiction Level'
])
df['Avg Time per Session'] = df['Total Time Spent'] / df['Number of Sessions']
df['Hookedness Score'] = df['Engagement'] * df['Scroll Rate']

features = [
    'Platform',
    'Total Time Spent',
    'Number of Sessions',
    'Engagement',
    'Scroll Rate',
    'Avg Time per Session',
    'Hookedness Score'
]
target = 'Addiction Level'

X = df[features]
y = df[target]

# Preprocessor
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), ['Platform'])
], remainder='passthrough')

# Pipeline
model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])

# Train and save
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

with open("addiction_model_v4.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model v4 saved as addiction_model_v4.pkl")
