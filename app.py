# Import necessary libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Load the data
data = pd.read_csv('2024 Ballon Dor Nominees League Stats.csv')

# Filter the data for Rodri and Vinicius Junior
players = data[data['player'].isin(['Rodri', 'Vinicius Júnior'])]

# Select relevant features for prediction
features = ['Performance-Gls', 'Performance-Ast', 'Expected-xG', 'Expected-xAG', 'Progression-PrgC', 'Progression-PrgP']

# Prepare the feature matrix X and target vector y
X = players[features]
y = players['player'] # Use player names as the target

# Scale the feature to normalize their values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create and train the Random Forest Model
model = RandomForestClassifier(n_estimators = 100, random_state = 42)
model.fit(X_scaled, y)

# Make predictions for both players
predictions = model.predict(X_scaled)

# Get feature importances
importances = model.feature_importances_

# Print the winner and feature importances
winner = predictions[0] # Since we only have 2 players, we can take the first prediction
print(f"The player more likely to win the Ballon d'Or between Vini and Rodri is: {winner}")

print("\nFeature Importances:")
for feature, importance in zip(features, importances):
    print(f"{feature}: {importance:.4f}")