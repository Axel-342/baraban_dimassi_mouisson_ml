import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Charger le fichier CSV
file_2023 = "atp_matches_qual_chall_2023.csv"
df_2023 = pd.read_csv(file_2023)

# Garder uniquement les colonnes demandées (sans winner_ioc)
filtered_df = df_2023[['winner_hand', 'winner_ht', 'winner_age']]

# Combler les valeurs manquantes dans 'winner_ht' par la moyenne de cette colonne
filtered_df['winner_ht'] = filtered_df['winner_ht'].fillna(filtered_df['winner_ht'].mean())

# Sauvegarder le fichier traité
output_file = "winner_profile_2023.csv"
filtered_df.to_csv(output_file, index=False)

print(f"Le fichier traité a été sauvegardé ici : {output_file}")

# Encoder les données catégoriques
le_hand = LabelEncoder()
filtered_df['winner_hand'] = le_hand.fit_transform(filtered_df['winner_hand'].fillna("Unknown"))

# Préparer les données
X = filtered_df[['winner_hand', 'winner_ht', 'winner_age']]
y = filtered_df['winner_hand']  # La main dominante comme cible pour un profil typique

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraîner le modèle Random Forest
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Prédire sur l'ensemble de test
y_pred = model.predict(X_test)

# Générer le profil type pour 2024
profile_2024 = pd.DataFrame({
    'winner_hand': [le_hand.inverse_transform([model.predict([[0, X['winner_ht'].mean(), X['winner_age'].mean()]])[0]])[0]],
    'winner_ht': [X['winner_ht'].mean()],
    'winner_age': [X['winner_age'].mean()]
})

# Affichage du profil type
print("Profil type du joueur parfait pour 2024 :")
print(profile_2024)

# Sauvegarder le profil type
profile_output_file = "winner_profile_2024_prediction.csv"
profile_2024.to_csv(profile_output_file, index=False)

print(f"Le profil type a été sauvegardé ici : {profile_output_file}")

# Créer un graphique avec matplotlib
plt.figure(figsize=(8, 6))

# Données à afficher dans le graphique
labels = ['Winner Hand', 'Winner Height', 'Winner Age']

# Conversion de winner_hand en chaîne pour l'affichage
winner_hand_str = str(profile_2024['winner_hand'].values[0])

# Liste des valeurs à afficher
values = [winner_hand_str, profile_2024['winner_ht'].values[0], profile_2024['winner_age'].values[0]]

# Créer un graphique à barres
plt.bar(labels, values, color=['blue', 'green', 'orange'])

# Ajouter un titre et des labels
plt.title("Profil du Joueur Parfait pour 2024")
plt.xlabel("Caractéristiques")
plt.ylabel("Valeurs")

# Afficher le graphique
plt.show()