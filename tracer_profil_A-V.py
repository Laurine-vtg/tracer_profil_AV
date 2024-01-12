import streamlit as st
import pandas as pd
import numpy as np 
import seaborn as sns
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Titre de l'app
st.markdown("<h1 style='text-decoration: underline;'>Tracer un profil A-V</h1>", unsafe_allow_html=True)

# Paramètres physiques
#nom = st.text_input("Nom Prénom")
#masse = st.number_input("Masse (kg)", value=0)
#taille_cm = st.number_input("Taille (cm)", value=0)
#taille = taille_cm / 100

# Charger les données à partir des fichiers CSV
uploaded_files = st.file_uploader("Choisissez un ou plusieurs fichiers CSV", type="csv", accept_multiple_files=True)
data_frames = []

if uploaded_files:
    for uploaded_file in uploaded_files:
        data = pd.read_csv(uploaded_file, delimiter=';', decimal=',')

# Extraire le nom du fichier CSV
        nom_fichier = uploaded_file.name 

        # Ne conserver que certaines colonnes
        colonnes_selectionnees = ['Seconds', 'Velocity', 'HDOP', ' #Sats']
        donnees_filtrees = data[colonnes_selectionnees]

        # Appliquer le filtre passe-bas de Butterworth
        if uploaded_file is not None:
            # Fréquence de coupure
            cutoff_frequency = 1  # en Hz

            # Fréquence d'échantillonnage
            sampling_frequency = 50.0  # en Hz

            # Ordre du filtre
            filter_order = 2

            # Calculer les coefficients du filtre
            b, a = butter(filter_order, cutoff_frequency / (sampling_frequency / 2), btype='low')

            # Appliquer le filtre aux données
            donnees_filtrees['Velocity_filtered'] = filtfilt(b, a, data['Velocity'])

        # Calculer l'accélération
        donnees_filtrees['Acceleration'] = donnees_filtrees['Velocity_filtered'].diff() / 0.02
        donnees_filtrees['Acceleration'] = donnees_filtrees['Acceleration'].fillna(0)

        # Ajouter le DataFrame traité à la liste
        data_frames.append(donnees_filtrees)
    
     # Tracer vitesse brute et vitesse filtrée en fonction du temps
        fig, ax = plt.subplots()
        ax.plot(donnees_filtrees['Seconds'], donnees_filtrees['Velocity'], label='Non filtré', linewidth=1)
        ax.plot(donnees_filtrees['Seconds'], donnees_filtrees['Velocity_filtered'], label='Filtré', linewidth=1)

    # Ajuster les limites des axes
        plt.xlim(0)
        plt.ylim(0)

        ax.set_title(nom_fichier)
        ax.set_xlabel('Temps (s)')
        ax.set_ylabel('Vitesse (m/s)')
        ax.legend()
        st.pyplot(fig)
        st.write("La fréquence de coupure est de 1Hz et la fréquence d'échantillonnage est de 50Hz.")

# Vérifier si les données sont chargées
        if 'Velocity_filtered' in donnees_filtrees.columns and 'Acceleration' in donnees_filtrees.columns:
    # Tracer vitesse filtrée et accélération en fonction du temps sur le même graphique
             fig, ax = plt.subplots(figsize=(10, 6))

    # Plot de la vitesse filtrée en fonction du temps
             ax.plot(donnees_filtrees['Seconds'], donnees_filtrees['Velocity_filtered'], label='Vitesse Filtrée', color='blue', linewidth=1)

    # Plot de l'accélération en fonction du temps
             ax.plot(donnees_filtrees['Seconds'], donnees_filtrees['Acceleration'], label='Accélération', color='red', linewidth=1)

# Ajuster les limites des axes
             plt.xlim(0)

             ax.set_title(nom_fichier)
             ax.set_xlabel('Temps (s)')
             ax.set_ylabel('Vitesse (m/s) / Accélération (m/s²)')
             ax.legend()



             st.pyplot(fig)
        else:
             st.warning("Assurez-vous que les colonnes Velocity_filtered et Acceleration sont présentes dans le fichier CSV chargé.")

    # Concaténer les DataFrames en un seul
    consolidated_data = pd.concat(data_frames, ignore_index=True)

else:
    st.warning("Veuillez charger un ou plusieurs fichiers CSV.")



# Calculer la moyenne des colonnes 'HDOP' et '#Sats'
moyenne_hdop = consolidated_data['HDOP'].mean()
moyenne_sats = consolidated_data[' #Sats'].mean()

# Afficher les moyennes
st.write('<span style="color: blue;">Qualité du signal :</span>', unsafe_allow_html=True)
st.write(f"En moyenne la dispersion des satellites (HDOP) vaut : {round(moyenne_hdop, 2)}")
st.write(f"Le nombre moyen de satellites est de : {round(moyenne_sats, 2)}")







# Afficher le tableau des données de vitesse filtrées
st.write('<span style="color: blue;">Tableau des données de vitesse filtrée et d\'accélération :</span>', unsafe_allow_html=True)
st.write(consolidated_data[['Seconds', 'Velocity', 'HDOP', ' #Sats', 'Velocity_filtered', 'Acceleration']])

# Vérifier si les données sont chargées
if 'Velocity_filtered' in consolidated_data.columns and 'Acceleration' in consolidated_data.columns:
    # Filtrer uniquement les valeurs positives
    donnees_positives = consolidated_data[consolidated_data['Acceleration'] > 0]

    # Filtrer les points avec une vitesse inférieure à 3 m/s
    donnees_basse_vitesse = donnees_positives[donnees_positives['Velocity_filtered'] < 3]

    # Créer le nuage de points avec les données positives
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot des points avec vitesse inférieure à 3 m/s en gris
    ax.scatter(donnees_basse_vitesse['Velocity_filtered'], donnees_basse_vitesse['Acceleration'], color='grey', s= 2)

    # Plot des points avec vitesse supérieure ou égale à 3 m/s en noir
    ax.scatter(donnees_positives[donnees_positives['Velocity_filtered'] >= 3]['Velocity_filtered'],
               donnees_positives[donnees_positives['Velocity_filtered'] >= 3]['Acceleration'], color='black', s= 2)

# Ajuster les limites des axes
    plt.xlim(0)
    plt.ylim(0)

    ax.set_title('Profil A-V')
    ax.set_xlabel('Vitesse (m/s)')
    ax.set_ylabel('Accélération (m/s²)')
    ax.legend()
    st.pyplot(fig)
else:
    st.warning("Assurez-vous que les colonnes Velocity et Acceleration sont présentes dans le fichier CSV chargé.")



# Identifier la vitesse maximale
vitesse_maximale = consolidated_data['Velocity_filtered'].max()

# Créer des intervalles de vitesse
intervalles_vitesse = np.arange(3, vitesse_maximale + 0.2, 0.2)

# Catégoriser les vitesses dans les intervalles et obtenir les deux accélérations maximales pour chaque intervalle
consolidated_data['Intervalle_Vitesse'] = pd.cut(consolidated_data['Velocity_filtered'], bins=intervalles_vitesse)

# Obtenir les deux valeurs d'accélération maximale pour chaque intervalle
acceleration_max_par_intervalle = consolidated_data.groupby('Intervalle_Vitesse')['Acceleration'].nlargest(2).reset_index(level=0, drop=True)

# Créer un nouveau DataFrame avec les valeurs d'accélération maximales
resultat_df = consolidated_data.loc[acceleration_max_par_intervalle.index, ['Velocity_filtered', 'Acceleration']]

# Afficher ou visualiser la nouvelle table
st.write('<span style="color: blue;">Tableau des deux valeurs d\'accélération maximale par intervalle de vitesse de 0.2 m/s (à partir de 3 m/s) :</span>', unsafe_allow_html=True)
st.write(resultat_df)








# Créer le nuage de points avec les données positives
fig, ax = plt.subplots(figsize=(10, 6))

# Plot des points avec vitesse inférieure à 3 m/s en gris
ax.scatter(donnees_basse_vitesse['Velocity_filtered'], donnees_basse_vitesse['Acceleration'], color='grey', s= 2)

# Plot des points avec vitesse supérieure ou égale à 3 m/s en noir
ax.scatter(donnees_positives[donnees_positives['Velocity_filtered'] >= 3]['Velocity_filtered'],
           donnees_positives[donnees_positives['Velocity_filtered'] >= 3]['Acceleration'], color='black', s= 2)

sns.scatterplot(x='Velocity_filtered', y='Acceleration', data=resultat_df, color='blue', s= 40)

# Calculer la régression linéaire avec Statsmodels
X = sm.add_constant(resultat_df['Velocity_filtered'])  # Ajouter une colonne constante
y = resultat_df['Acceleration']
model = sm.OLS(y, X).fit()

# Obtenir les paramètres de régression
coef_estimation = model.params[1]
intercept_estimation = model.params[0]

# Calculer l'intervalle de confiance pour la pente (coefficient)
confidence_interval = model.conf_int(alpha=0.05, cols=None)
st.write (confidence_interval)


# Obtenir l'équation de la droite de régression
equation = f"y = {round(model.params[1], 2)} * x + {round(model.params[0], 2)}"
   
# Obtenir le coefficient de détermination (R²)
r_squared = f"{round(model.rsquared, 2)}"

# Trouver les coordonnées des points d'intersection avec les axes
intercept_x = -model.params[0] / model.params[1]
intercept_x_kmh = intercept_x * 3.6
intercept_y = model.params[0]


# Tracer une droite reliant les points d'intersection
plt.plot([intercept_x, 0], [0, intercept_y], linestyle='-', color='blue')

# Ajuster les limites des axes
plt.xlim(0)
plt.ylim(0)

# titre
ax.set_title('Profil A-V')
ax.set_xlabel('Vitesse (m/s)')
ax.set_ylabel('Accélération (m/s²)')
ax.legend()
st.pyplot(fig)

st.write('<span style="color: blue;">Résultats avant la supression des valeurs aberrantes hors de l\'intervalle de confiance :</span>', unsafe_allow_html=True)

# Afficher l'équation de la droite et le coefficient de détermination
st.write(f'Équation de la droite : {equation}')
st.write(f'Coefficient de détermination (R²) : {r_squared}')

#inscrire la valeur de V0 et A0
st.write(f"S0 : {round(intercept_x,2)} m/s")
st.write(f"S0 : {round(intercept_x_kmh,2)} km/h")
st.write(f"A0 : {round(intercept_y,2)} m/s²")

# Tracer la régression linéaire avec l'intervalle de confiance
fig, ax = plt.subplots(figsize=(10, 6))

ax.scatter(donnees_basse_vitesse['Velocity_filtered'], donnees_basse_vitesse['Acceleration'], color='grey', s= 2)
ax.scatter(donnees_positives[donnees_positives['Velocity_filtered'] >= 3]['Velocity_filtered'],
           donnees_positives[donnees_positives['Velocity_filtered'] >= 3]['Acceleration'], color='black', s= 2)
sns.scatterplot(x='Velocity_filtered', y='Acceleration', data=resultat_df, color='blue', s= 40)

# Tracer la droite de régression
ax.plot(resultat_df['Velocity_filtered'], model.predict(X), label='Régression linéaire', color='red')

# Tracer l'intervalle de confiance
ax.fill_between(resultat_df['Velocity_filtered'], confidence_interval[0][1] * resultat_df['Velocity_filtered'] + confidence_interval[0][0],
                confidence_interval[1][1] * resultat_df['Velocity_filtered'] + confidence_interval[1][0], color='red', alpha=0.2)


# Ajuster les limites des axes
plt.xlim(0)
plt.ylim(0)

# titre
ax.set_title('Profil A-V avec Régression Linéaire et Intervalle de Confiance')
ax.set_xlabel('Vitesse (m/s)')
ax.set_ylabel('Accélération (m/s²)')
ax.legend()
st.pyplot(fig)


# Afficher l'intervalle de confiance pour la pente (coefficient)
#st.write('Intervalle de confiance pour la pente (coefficient) :')
#st.write(confidence_interval)


# Calculer les bornes de l'intervalle de confiance pour les valeurs d'accélération prédites
lower_bound = confidence_interval[0][1] * resultat_df['Velocity_filtered'] + confidence_interval[0][0]
upper_bound = confidence_interval[1][1] * resultat_df['Velocity_filtered'] + confidence_interval[1][0]

# Vérifier si les valeurs d'accélération dans resultat_df sont dans l'intervalle de confiance
resultat_df['Dans_Intervalle_Confiance'] = (resultat_df['Acceleration'] >= lower_bound) & (resultat_df['Acceleration'] <= upper_bound)

# Afficher le dataframe mis à jour avec la colonne 'Dans_Intervalle_Confiance'
st.write(resultat_df)

# Compter le nombre de valeurs en dehors de l'intervalle de confiance
valeurs_hors_intervalle = resultat_df[resultat_df['Dans_Intervalle_Confiance'] == False].shape[0]

# Afficher le nombre de valeurs en dehors de l'intervalle de confiance
st.write(f"Nombre de valeurs en dehors de l'intervalle de confiance : {valeurs_hors_intervalle}")

# Afficher le pourcentage de valeurs en dehors de l'intervalle de confiance
#pourcentage_hors_intervalle = (valeurs_hors_intervalle / resultat_df.shape[0]) * 100
#st.write(f"Pourcentage de valeurs en dehors de l'intervalle de confiance : {round(pourcentage_hors_intervalle,2)}%")

# Créer un nouveau dataframe en ne conservant que les lignes dans l'intervalle de confiance
resultat_df_dans_intervalle = resultat_df.loc[resultat_df['Dans_Intervalle_Confiance'], ['Velocity_filtered', 'Acceleration']]

# Afficher le nouveau dataframe
#st.write('<span style="color: blue;">Nouveau tableau des valeurs dans l\'intervalle de confiance :</span>', unsafe_allow_html=True)
#st.write(resultat_df_dans_intervalle)









# Créer le nuage de points avec les données positives
fig, ax = plt.subplots(figsize=(10, 6))

# Plot des points avec vitesse inférieure à 3 m/s en gris
ax.scatter(donnees_basse_vitesse['Velocity_filtered'], donnees_basse_vitesse['Acceleration'], color='grey', s = 2)

# Plot des points avec vitesse supérieure ou égale à 3 m/s en noir
ax.scatter(donnees_positives[donnees_positives['Velocity_filtered'] >= 3]['Velocity_filtered'],
           donnees_positives[donnees_positives['Velocity_filtered'] >= 3]['Acceleration'], color='black', s = 2)

sns.scatterplot(x='Velocity_filtered', y='Acceleration', data=resultat_df_dans_intervalle, color='blue', s=40)

# Calculer la régression linéaire avec Statsmodels
X = sm.add_constant(resultat_df_dans_intervalle['Velocity_filtered'])  # Ajouter une colonne constante
y = resultat_df_dans_intervalle['Acceleration']
model = sm.OLS(y, X).fit()

# Obtenir les paramètres de régression
coef_estimation = model.params[1]
intercept_estimation = model.params[0]

# Calculer l'intervalle de confiance pour la pente (coefficient)
confidence_interval = model.conf_int(alpha=0.05, cols=None)


# Obtenir l'équation de la droite de régression
equation_final = f"y = {round(model.params[1], 2)} * x + {round(model.params[0], 2)}"
   
# Obtenir le coefficient de détermination (R²)
r_squared_final = f"{round(model.rsquared, 2)}"

# Trouver les coordonnées des points d'intersection avec les axes
intercept_x = -model.params[0] / model.params[1]
intercept_x_kmh = intercept_x * 3.6
intercept_y = model.params[0]


# Tracer une droite reliant les points d'intersection
plt.plot([intercept_x, 0], [0, intercept_y], linestyle='-', color='blue')

# Ajuster les limites des axes
plt.xlim(0)
plt.ylim(0)

# titre
ax.set_title('Profil A-V final')
ax.set_xlabel('Vitesse (m/s)')
ax.set_ylabel('Accélération (m/s²)')
ax.legend()
st.pyplot(fig)

st.write('<span style="color: blue;">Résultats finaux :</span>', unsafe_allow_html=True)
# Afficher l'équation de la droite et le coefficient de détermination
st.write(f'Équation de la droite : {equation_final}')
st.write(f'Coefficient de détermination (R²) : {r_squared_final}')

#inscrire la valeur de V0 et A0
st.write(f"S0 : {round(intercept_x,2)} m/s")
st.write(f"S0 : {round(intercept_x_kmh,2)} km/h")
st.write(f"A0 : {round(intercept_y,2)} m/s²")

#calcul pente 
pente = - intercept_y / intercept_x 
st.write(f"Pente : {round(pente,2)}")

#nombre de points pour faire la regression
nombre_points_acc = resultat_df_dans_intervalle.shape[0]
st.write(f"Nombre de points utilisés pour tracer la regression : {round(nombre_points_acc)}")

#connaitre le nombre de points brutes au départ
points_bruts = consolidated_data.shape[0]
st.write(f"Nombre de points de données brutes : {points_bruts} ")

#determiner Vmax
Vmax = consolidated_data['Velocity_filtered'].max()
st.write(f"Vmax : {round(Vmax,2)} m/s")
