import pandas as pd

# Chemin des fichiers CSV
file_2024 = "atp_matches_qual_chall_2024.csv"
file_2023 = "atp_matches_qual_chall_2023.csv"

# Lire les fichiers en DataFrames
df_2024 = pd.read_csv(file_2024)
df_2023 = pd.read_csv(file_2023)

# Concaténer les deux DataFrames
merged_df = pd.concat([df_2024, df_2023])

# Garder seulement les colonnes winner_age et winner_id
filtered_df = merged_df[['winner_age', 'winner_id']]

# Sauvegarder le résultat dans un fichier CSV
output_file = "merged_winner_data.csv"
filtered_df.to_csv(output_file, index=False)
print(f"Le fichier fusionné a été sauvegardé ici : {output_file}")