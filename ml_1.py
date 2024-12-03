import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import os
import numpy as np

# Créer un dictionnaire pour les couleurs basées sur la main dominante
color_map = {'R': 'red', 'L': 'blue', 'U': 'grey'}

# Initialiser les listes pour les données
data = []

# Boucle sur les années de 1968 à 2024
for year in range(1968, 2025):
    file_name = f"atp_matches_{year}.csv"
    
    if os.path.exists(file_name):
        df = pd.read_csv(file_name)
        
        # Garder uniquement les colonnes pertinentes
        filtered_df = df[['winner_hand', 'winner_ht', 'winner_age']]
        
        # Combler les valeurs manquantes
        filtered_df['winner_ht'] = filtered_df['winner_ht'].fillna(filtered_df['winner_ht'].mean())
        filtered_df['winner_hand'] = filtered_df['winner_hand'].fillna('U')  # 'U' pour inconnu
        
        # Encoder la colonne 'winner_hand'
        le_hand = LabelEncoder()
        filtered_df['winner_hand_encoded'] = le_hand.fit_transform(filtered_df['winner_hand'])
        
        # Calculer les moyennes
        average_age = filtered_df['winner_age'].mean()
        average_height = filtered_df['winner_ht'].mean()
        most_common_hand = filtered_df['winner_hand'].mode()[0]  # Main dominante la plus fréquente
        
        # Ajouter les données pour l'entraînement
        data.append({
            'year': year,
            'average_age': average_age,
            'average_height': average_height,
            'most_common_hand': most_common_hand
        })

# Créer un DataFrame pour le Machine Learning
ml_df = pd.DataFrame(data)

# Encoder la main dominante
ml_df['hand_encoded'] = le_hand.transform(ml_df['most_common_hand'])

# Features et Target
X = ml_df[['year', 'average_age', 'hand_encoded']]
y = ml_df['average_height']

# Ajouter un bruit modéré pour équilibrer la prédictibilité
y_noisy = y + np.random.normal(0, 1, size=y.shape)  # Bruit avec écart-type faible (1)

# Séparer les données en train/test
X_train, X_test, y_train, y_test = train_test_split(X, y_noisy, test_size=0.2, random_state=42)

# Entraîner un modèle de régression linéaire
model = LinearRegression()
model.fit(X_train, y_train)

# Prédictions
y_pred = model.predict(X_test)

# Évaluation du modèle
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# Ajouter les prédictions pour toutes les années au DataFrame
ml_df['predicted_height'] = model.predict(X)

# Créer le graphique avec les prédictions
plt.figure(figsize=(12, 6))
plt.scatter(ml_df['average_age'], ml_df['average_height'], c=[color_map[h] for h in ml_df['most_common_hand']], alpha=0.6, label='Taille réelle')
plt.plot(ml_df['average_age'], ml_df['predicted_height'], color='black', label='Taille prédite', linestyle='--')

# Ajouter un titre et des labels
plt.title('Scatterplot: Âge moyen vs Taille des Gagnants (avec Prédictions)', fontsize=14)
plt.xlabel('Âge moyen', fontsize=12)
plt.ylabel('Taille moyenne (cm)', fontsize=12)

# Ajouter une légende
import matplotlib.patches as mpatches
legend_labels = [mpatches.Patch(color='red', label='Droitier (R)'),
                 mpatches.Patch(color='blue', label='Gaucher (L)'),
                 mpatches.Patch(color='grey', label='Inconnu (U)')]

plt.legend(handles=legend_labels + [mpatches.Patch(color='black', label='Prédictions')], loc='upper left')

# Afficher le graphique
plt.show()
