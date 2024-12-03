import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import os

# Créer un dictionnaire pour les couleurs basées sur la main dominante
color_map = {'R': 'red', 'L': 'blue', 'U': 'grey'}

# Initialiser les listes pour les données à tracer
years = []
ages = []
heights = []
colors = []

# Boucle sur les années de 1968 à 2024
for year in range(1968, 2025):
    # Charger les données de l'année correspondante
    file_name = f"atp_matches_{year}.csv"
    
    # Vérifier si le fichier existe avant de le traiter
    if os.path.exists(file_name):
        df = pd.read_csv(file_name)
        
        # Garder uniquement les colonnes pertinentes
        filtered_df = df[['winner_hand', 'winner_ht', 'winner_age']]
        
        # Combler les valeurs manquantes dans 'winner_ht' par la moyenne de cette colonne
        filtered_df['winner_ht'] = filtered_df['winner_ht'].fillna(filtered_df['winner_ht'].mean())
        
        # Encoder la colonne 'winner_hand' pour déterminer les couleurs
        le_hand = LabelEncoder()
        filtered_df['winner_hand_encoded'] = le_hand.fit_transform(filtered_df['winner_hand'].fillna("U"))
        
        # Calculer les valeurs moyennes pour chaque année
        average_age = filtered_df['winner_age'].mean()
        average_height = filtered_df['winner_ht'].mean()
        most_common_hand = filtered_df['winner_hand'].mode()[0]  # Main dominante la plus fréquente
        
        # Ajouter les valeurs pour le graphique
        years.append(year)
        ages.append(average_age)
        heights.append(average_height)
        colors.append(color_map.get(most_common_hand, 'grey'))  # Assigner la couleur de la main dominante

print(ages)

# Créer le graphique scatterplot
plt.figure(figsize=(12, 6))
plt.scatter(ages, heights, c=colors, alpha=0.6)

# Ajouter un titre et des labels
plt.title('Scatterplot: Année vs Taille des Gagnants (1978-2024)', fontsize=14)
plt.xlabel('Année', fontsize=12)
plt.ylabel('Taille (cm)', fontsize=12)

# Ajouter une légende manuelle pour les couleurs
import matplotlib.patches as mpatches
legend_labels = [mpatches.Patch(color='red', label='Droitier (R)'),
                 mpatches.Patch(color='blue', label='Gaucher (L)'),
                 mpatches.Patch(color='grey', label='Inconnu (U)')]

plt.legend(handles=legend_labels, loc='upper left')


# Afficher le graphique
plt.show()

