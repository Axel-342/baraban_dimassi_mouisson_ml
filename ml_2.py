# Importations nécessaires
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import os
import numpy as np

# Initialiser une liste pour stocker les données
data = []

# Parcourir les fichiers de 1968 à 2024
for year in range(1968, 2025):
    file_name = f"atp_matches_{year}.csv"
    
    if os.path.exists(file_name):
        df = pd.read_csv(file_name)
        
        # Garder uniquement les colonnes nécessaires
        filtered_df = df[['surface', 'winner_age', 'winner_hand', 'winner_ht', 'w_svpt', 'w_1stIn', 'w_1stWon', 'w_ace']]
        
        # Gérer les valeurs manquantes
        filtered_df['winner_ht'] = filtered_df['winner_ht'].fillna(filtered_df['winner_ht'].mean())
        filtered_df['winner_hand'] = filtered_df['winner_hand'].fillna('U')  # Main inconnue
        filtered_df['surface'] = filtered_df['surface'].fillna('Unknown')
        filtered_df = filtered_df.dropna()  # Suppression des lignes restantes avec des valeurs manquantes
        
        # Encoder les colonnes catégoriques
        le_surface = LabelEncoder()
        le_hand = LabelEncoder()
        filtered_df['surface_encoded'] = le_surface.fit_transform(filtered_df['surface'])
        filtered_df['winner_hand_encoded'] = le_hand.fit_transform(filtered_df['winner_hand'])
        
        # Ajouter les données dans la liste principale
        data.append(filtered_df)

# Concaténer toutes les données en un seul DataFrame
final_df = pd.concat(data, ignore_index=True)

# Définir les Features (X) et la Target (y)
X = final_df[['winner_age', 'winner_ht', 'w_svpt', 'w_1stIn', 'w_1stWon', 'surface_encoded', 'winner_hand_encoded']]
y = final_df['w_ace']

# Ajouter un peu de bruit pour une meilleure généralisation
y_noisy = y + np.random.normal(0, 1, size=y.shape)

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y_noisy, test_size=0.2, random_state=42)

# Créer et entraîner le modèle de régression linéaire
model = LinearRegression()
model.fit(X_train, y_train)

# Prédire sur l'ensemble de test
y_pred = model.predict(X_test)

# Évaluer les performances
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# Visualisation des résultats : Réel vs Prédit
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.6, label='Prédictions')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label='Ligne parfaite')
plt.title("Prédictions vs Valeurs Réelles des Aces", fontsize=14)
plt.xlabel("Nombre réel d'aces", fontsize=12)
plt.ylabel("Nombre prédit d'aces", fontsize=12)
plt.legend()
plt.show()

# Visualisation des moyennes d'aces par surface
final_df['total_aces'] = final_df['w_ace']
surface_means = final_df.groupby('surface')['total_aces'].mean()

plt.figure(figsize=(12, 6))
surface_means.plot(kind='bar', color='skyblue')
plt.title("Nombre moyen d'aces par match selon la surface", fontsize=14)
plt.xlabel("Surface", fontsize=12)
plt.ylabel("Nombre moyen d'aces", fontsize=12)
plt.xticks(rotation=45)
plt.show()
