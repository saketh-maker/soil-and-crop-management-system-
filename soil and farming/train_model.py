# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Sample data (Replace with actual data)
data = {
    'soil_type': [1, 2, 3, 1, 2, 3],
    'region': [1, 2, 3, 1, 2, 3],
    'crop': ['Wheat', 'Peanuts', 'Rice', 'Corn', 'Wheat', 'Peanuts']
}

df = pd.DataFrame(data)

X = df[['soil_type', 'region']]
y = df['crop']

# Convert categorical values to numeric
X = pd.get_dummies(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'model.pkl')
