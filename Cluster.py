# -*- coding: utf-8 -*-
"""
Clusteranalsen auf Gamerdaten
Created on Thu Jun 14 09:24:38 2018

@author: H.Baars
"""

import pandas as pd
import matplotlib.pyplot as plt
import sklearn.cluster as skc
import numpy as np
from scipy.cluster.hierarchy import dendrogram

def plot_dendrogram(model, **kwargs):

    # Children of hierarchical clustering
    children = model.children_

    # Distances between each pair of children
    # Since we don't have this information, we can use a uniform one for plotting
    distance = np.arange(children.shape[0])

    # The number of observations contained in each cluster level
    no_of_observations = np.arange(2, children.shape[0]+2)

    # Create linkage matrix and then plot the dendrogram
    linkage_matrix = np.column_stack([children, distance, no_of_observations]).astype(float)
    print('Kuckuck!')
    # Plot the corresponding dendrogram
    
    fig.clear()
    ax = fig.add_subplot(111)
    ax.legend = True
    dendrogram(linkage_matrix, color_threshold=392, no_labels=True, ax=ax) #p=100, truncate_mode='lastp', 
    ax.add_title='Dendrogramm für Agglomeratives Clustering (Average Linking)'
    ax.axes.yaxis.set_label='Clustergröße'
    plt.show()
    
#1. Daten einlesen und in DataFrame df_CA speichern
fileCA = r'C:\Users\ac104570\Documents\SoSe18Vo\Analytics\DataSets\GamerToCluster.csv'
df_CA_KM = pd.read_csv(fileCA, sep=';', encoding='utf-8')

#2. Daten standardisieren, Maxima zur Rekonstruktion in altMax und UmsatzMax merken!
altMax = df_CA_KM.Alter.max()
UmsatzMax = df_CA_KM.Umsatz.max()
df_CA_KM.Alter = df_CA_KM.Alter / altMax
df_CA_KM.Umsatz = df_CA_KM.Umsatz / UmsatzMax

#3. Selektion der relevanten Spalten für die eigentliche Analyse, hinterlegt in df_CAS
# Hinweis: die Methode loc() erlaubt es, bequuem mehrere Zeilen & Spalten nach Name auszuwählen
#          mit iloc() können Zeilen- und Spaltennummern genutzt werden
df_CAS = df_CA_KM.loc[:,'Alter':'Umsatz']

#4. Modell initialisieren und auf die Daten loslossen
cluster = skc.KMeans(6)
cluster.fit(df_CAS)

#5. Ergebnisse: Centroiden in eigenes DataFrame, Clusternummern in df_CA_KM.cluster
centroids = pd.DataFrame(cluster.cluster_centers_)
df_CA_KM['cluster'] = cluster.predict(df_CAS) # cluster.labels_

#6. Liste mit Farbcodierungen für die Cluster,0 für rot, 1 für grün usw.
#   Farben in Spalte farbeKM von df_CA_KM hinterlegen
farbe = ['red', 'green', 'yellow', 'blue', 'black', 'orange', 'grey']
farbe2 = ['black', 'yellow', 'green', 'blue', 'red', 'orange', 'grey']
farbe3 = ['red', 'green', 'yellow', 'blue', 'black', 'orange', 'grey']
farbe4 = ['red', 'green', 'grey', 'black', 'blue', 'orange', 'yellow']

df_CA_KM['farbeKM'] =[farbe3[x+1] for x in df_CA_KM['cluster']]

#7. und jetzt plotten
ax = df_CA_KM.plot(kind='scatter', x='Alter', y = 'Umsatz', c = df_CA_KM['farbeKM'], figsize=(10,8))
centroids.plot(kind='scatter', x=0, y=1, color='red', s=100, marker='x', ax=ax)


#8. Elbow-Krierium ausprobiert: 
#   Methode für unterschiedliche Clusterzahlen wiederholen, jeweils den
#   WSS(within cluster sum of squares) über Eigenschaft score abrufen und in guete hinterlegen
guete =list(range(0,9))
for i in range(1,10):
    cluster = skc.KMeans(i)
    cluster.fit(df_CAS)
    guete[i-1] = -cluster.score(df_CAS)

# 9. Elbow-Graph plotten
pd.DataFrame(guete).plot(figsize=(10,8))
# optional: speichern der Grafik mit savefig: 
#plt.savefig(r'C:\Users\ac104570\Documents\SoSe18Vo\Analytics\DataSets\elbow.png',type='png')

#10. Clusteranalyse mit DBScan, analog zu den Schritten 4-6
df_CA_DBS = df_CA_KM[:] #hier kopieren wir uns die Ausgangsdaten
cluster = skc.DBSCAN(eps=0.1, min_samples=10)
cluster.fit(df_CAS)
df_CA_DBS['cluster'] = cluster.labels_+1
df_CA_DBS['farbeDBS'] =[farbe4[x] for x in df_CA_DBS['cluster']]
    
#11. Clusteranalyse mit agglomerative Clustering, Kriterium AverageLinkage
#    (minimaler durchschnittlicher Abstand zwishcen den Punkten zweier Cluster als 
#    Verschmelzungskrierium)
df_CA_AGG_AL = df_CA_KM[:]
cluster = skc.AgglomerativeClustering(6, linkage='average')
cluster.fit(df_CAS)
df_CA_AGG_AL['cluster'] = cluster.labels_
df_CA_AGG_AL['farbeAL'] =[farbe[x] for x in df_CA_AGG_AL['cluster']]

#11. Clusteranalyse mit agglomerative Clustering, Kriterium Ward
#    (minimale Zunahme des WSS als Verschmelzungskritrium)
df_CA_AGG_WARD = df_CA_KM[:]
cluster = skc.AgglomerativeClustering(6, linkage='ward')
cluster.fit(df_CAS)
df_CA_AGG_WARD['cluster'] = cluster.labels_
df_CA_AGG_WARD['farbeWA'] =[farbe2[x] for x in df_CA_AGG_WARD['cluster']]


#12. Matplotlib sagen, dass jetzt eine Graphik fig mit 4 Untergraphiken kommt,
#    die in 2 Reihen und 2 Spalten angeordnet sind, sich die Achsen teilen und deren
#    Achsenobjekte ax1 - ax4 sind.
#    Damit dann die 4 Graohiken plotten
fig, ((ax1, ax2))  = plt.subplots(nrows=2, ncols=1, figsize=(10,11))
#df_CA_AGG_AL.plot(ax=ax1, title='Agglomerative (Average Linkage)', kind='scatter', x='Alter', y = 'Umsatz', c = df_CA_AGG_AL['farbeAL'])
#df_CA_AGG_WARD.plot(ax=ax2, title='Agglomerative (Ward)', kind='scatter', x='Alter', y = 'Umsatz', c = df_CA_AGG_WARD['farbeWA'])
df_CA_KM.plot(ax=ax1, title='K-Means', kind='scatter', x='Alter', y = 'Umsatz', c = df_CA_KM['farbeKM'])
df_CA_DBS.plot(ax=ax2, title='DBSCAN', kind='scatter', x='Alter', y = 'Umsatz', c = df_CA_DBS['farbeDBS'])
plt.subplots_adjust(hspace = 0.4)

# optional: speichern der Grafik mit savefig:
#plt.savefig(r'C:\Users\ac104570\Documents\SoSe18Vo\Analytics\DataSets\cluster.png',type='png')

#13. Pivot-Table zur weiteren Analyse der Cluster mit einem Barchart
PT=df_CA_DBS.loc[:,['cluster','Typ','Alter','Umsatz']]
PT['Alter']=PT['Alter']*altMax
PT['Umsatz']=PT['Umsatz']*UmsatzMax
PT.pivot_table(index='cluster', values=['Alter'], aggfunc='mean', columns='Typ').plot(kind='barh', figsize =(5,5))

df_CA_AGG_AL = df_CA_KM[:]
cluster = skc.AgglomerativeClustering(6, linkage='average')
cluster.fit(df_CAS)
#plot_dendrogram(cluster, ax=ax) #, labels=cluster.labels_
