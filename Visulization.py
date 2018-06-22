from sklearn.cluster import KMeans
import numpy as np
from numpy import linalg as la
from matplotlib import cm
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import metrics
from sklearn import preprocessing
from sklearn.decomposition import PCA, FastICA
import time

players_dataframe = pd.read_csv("nba_player_whole_stats.csv")
players_dataframe.drop(players_dataframe[players_dataframe.Games < 15].index, inplace=True)
# write data columns to file
# with open('column_dictionary.txt','w+') as f:
#     for item in players_dataframe.columns:
#         f.writelines(item + '\n')
# f.close()
# print players_dataframe.head()
print 'shape:', players_dataframe.shape #452x124, 411x124 after remove <15 games
players_dataframe.fillna(value=0.0, inplace=True)
# print players_dataframe.And_One
# missing_values_count = players_dataframe.isnull().sum()
# print type(missing_values_count)
# print missing_values_count
# print len(players_dataframe.loc[players_dataframe.And_One.isnull()].Player)


# print players_data.columns
players_name = players_dataframe.values[:,1] # np array
players_stat = players_dataframe.values[:,3:] # np array
players_stat_columns = players_dataframe.columns.values[3:]
num_players, num_features  = players_stat.shape

min_max_scaler = preprocessing.MinMaxScaler()
player_stat_normalized = min_max_scaler.fit_transform(players_stat)
players_stat_normalized_df = pd.DataFrame(player_stat_normalized)


pca_17d_transformed_data = np.load( "nba_pca_transformed_17d_matrix.npy")
pca_3d_transformed_data = np.load( "nba_pca_transformed_3d_matrix.npy")
# ica_transformed_data = np.load( "ablone_ica_transformed_matrix.npy")
rp_transformed_data = np.load( "nba_rp_transformed_9d_matrix.npy")
d_reduction_alg_dict = {
    # 'Original': players_stat,
    'PCA_17d': pca_17d_transformed_data,
    'pca_3d': pca_3d_transformed_data,
    # 'ica': ica_transformed_data,
    'rp': rp_transformed_data}


colors = ['darkorchid', 'turquoise', 'darkorange', 'crimson', 'green',
          'dodgerblue', 'grey', 'greenyellow', 'navy', 'aqua',
          'brown', 'crimson', 'darkgoldenrod']

def kmeans_plot_pca_3d():
    cluster_results = np.load("k_means_k-means++_11clusters.npy")
    cluster_results = cluster_results[6] # pick the row of 11 clusters
    num_clusters=np.amax(cluster_results)+1
    print num_clusters
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for color, cluster in zip(colors, range(num_clusters)):
        ax.scatter(pca_3d_transformed_data[cluster_results == cluster, 0],
                   pca_3d_transformed_data[cluster_results == cluster, 1],
                   pca_3d_transformed_data[cluster_results == cluster, 2]
                   , c=color, label=cluster, marker='.')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title(('NBA Player Stats, %d clusters on pca 3d data') % num_clusters)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.show()

def kmeans_plot_clustering_names():
    cluster_results = np.load("k_means_k-means++_11clusters.npy")
    cluster_results = cluster_results[6] # pick the row of 11 clusters
    num_clusters=np.amax(cluster_results)+1
    for color, cluster in zip(colors, range(num_clusters)):
        print cluster
        print players_name[cluster_results == cluster]

# kmeans_plot_pca_3d()
# kmeans_plot_clustering_names()

def plot_boxplot():
    width = 3
    hight = 3
    step = 1 + num_features / (width * hight)
    print 'step', step
    # fig, ax = plt.subplots(width, hight)
    start = 0
    end = step
    plt.figure(figsize=(16, 9))
    for i in range(3):
        for j in range(3):
            idx = 330 + i*3+j+1
            plt.subplot(idx)
            plt.boxplot(player_stat_normalized[:, start:end], 0, '')
            plt.xticks(np.arange(step),players_stat_columns[start:end],rotation=30)
            plt.xticks(fontsize=6)
            start += step
            end += step
    plt.suptitle('Normalized Data', fontsize=16)
    plt.savefig('NormalizedData_BoxPlot.png')
    plt.close()
plot_boxplot()