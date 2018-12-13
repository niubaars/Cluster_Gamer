# -*- coding: utf-8 -*-
"""
3-Dimensionale Cluster-Analyse mit KMeans
Created on Thu Jun 14 09:24:38 2018

@author: Henning Baars
"""

# 0. Import und Daten einlesen wie gehabt
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.cluster as skc
fileCA = r'C:\\Users\ac104570\Documents\Buch\MiningDaten'
df_CA_KM = pd.read_csv(fileCA, sep=';', encoding='utf-8')

# 1. Daten standardisieren, Farbliste anlegen
farbe = ['cyan', 'green', 'yellow', 'blue', 'black', 'orange', 'grey']
altMax = df_CA_KM.Alter.max()
UmsatzMax = df_CA_KM.Umsatz.max()
FrequMax = df_CA_KM.Frequenz.max()
df_CA_KM.Alter = df_CA_KM.Alter / altMax
df_CA_KM.Umsatz = df_CA_KM.Umsatz / UmsatzMax
df_CA_KM.Frequenz = df_CA_KM.Frequenz / FrequMax

# 2. Daten selektieren, 6 Ckuster bilden
df_CAS = df_CA_KM.loc[:,['Alter','Umsatz','Frequenz']]
cluster = skc.KMeans(6)
cluster.fit(df_CAS)
centroids3D = pd.DataFrame(cluster.cluster_centers_)
df_CA_KM['cluster'] = cluster.predict(df_CAS) # cluster.labels_
df_CA_KM['farbeKM'] =[farbe[x+1] for x in df_CA_KM['cluster']]

# 3. Daten und Ergebnisse zurückskalieren
df_CA_KM.Umsatz *= UmsatzMax
df_CA_KM.Alter *= altMax
df_CA_KM.Frequenz *= FrequMax
centroids3D[0] *= altMax
centroids3D[1] *= UmsatzMax
centroids3D[2] *= FrequMax

# 4. 3D-Plot mit Matplotlib
# 4a. Größe setzen
fig = plt.figure()
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 10
fig_size[1] = 10
plt.rcParams["figure.figsize"] = fig_size

# 4b. 3D-lot aufsetzen und Scatterplot für Daten und Centroide
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xs=df_CA_KM['Alter'], ys = df_CA_KM['Umsatz'], zs=df_CA_KM['Frequenz'], c = df_CA_KM['farbeKM'])
ax.scatter(xs=centroids3D[0], ys = centroids3D[1], zs=centroids3D[2], marker='*', s=100, c = 'red')

# 4c. Centroide mit Stern-Symbolen einfügen
for i in centroids3D.T:
    ax.text(x=centroids3D.T[i][0], y=centroids3D.T[i][1], z=centroids3D.T[i][2], s='Centroid '+str(i), fontsize=18)

# 4c. Achsenbeschriftungen ergänzen
ax.set_xlabel('Alter')
ax.set_ylabel('Umsatz')
ax.set_zlabel('Frequenz')