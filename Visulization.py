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
from sklearn.metrics.cluster import homogeneity_score, completeness_score
import time

players_dataframe = pd.read_csv("nba_player_whole_stats.csv")
players_dataframe.drop(players_dataframe[players_dataframe.Games < 15].index, inplace=True)
players_dataframe.drop(columns='Total_Plus_Minus', inplace=True)

# write data columns to file
# with open('column_dictionary.txt','w+') as f:
#     for item in players_dataframe.columns:
#         f.writelines(item + '\n')
# f.close()
# print players_dataframe.head()
print 'shape:', players_dataframe.shape
players_dataframe.fillna(value=0.0, inplace=True)

for column in players_dataframe.filter(regex='Pct').columns:
    players_dataframe[column] = players_dataframe[column].apply(lambda x:x*100)

# print players_dataframe.filter(regex='Pct').head()

# print players_dataframe.And_One
# missing_values_count = players_dataframe.isnull().sum()
# print type(missing_values_count)
# print missing_values_count
# print len(players_dataframe.loc[players_dataframe.And_One.isnull()].Player)


# print players_data.columns
players_name = players_dataframe.values[:,1] # np array
players_stat = players_dataframe.values[:,3:] # np array #411x119 after remove <15 games
players_stat_columns = players_dataframe.columns.values[3:]
num_players, num_features  = players_stat.shape # (411, 119)

min_max_scaler = preprocessing.MinMaxScaler()
player_stat_normalized = min_max_scaler.fit_transform(players_stat)
players_stat_normalized_df = pd.DataFrame(player_stat_normalized)

### the following files include clustering lables from 5-80 clusters,
### each row represents result for a paticular value of k


### K-means labels ###
normalized_kmeans_clustering_np_matrix = np.load("Normalized_k_means_labels.npy")
original_kmeans_clustering_np_matrix = np.load("Original_k_means_labels.npy")
pca_kmeans_clustering_np_matrix = np.load("NBA_D_Reduction_PCA_17d_kmeans_plus_results.npy")
ica_original_kmeans_clustering_np_matrix = np.load("NBA_D_Reduction_ICA_original_13d_kmeans_plus_results.npy")
rp_original_kmeans_clustering_np_matrix = np.load("NBA_D_Reduction_RP_origininal_18d_kmeans_plus_results.npy")
rp_normalized_kmeans_clustering_np_matrix = np.load("NBA_D_Reduction_RP_normalized_17d_kmeans_plus_results.npy")

all_kmeans_clustering_np_matrix = {
'Normalized':    normalized_kmeans_clustering_np_matrix,
'Original':    original_kmeans_clustering_np_matrix,
'PCA':    pca_kmeans_clustering_np_matrix,
'ICA_original':    ica_original_kmeans_clustering_np_matrix,
# 'ICA_normalized':    ica_normalized_kmeans_clustering_np_matrix,
'RP_original':    rp_original_kmeans_clustering_np_matrix,
'RP_normalized':    rp_normalized_kmeans_clustering_np_matrix
}


### EM labels ###
normalized_em_clustering_np_matrix = np.load("NBA_EM_Normalized_results.npy")
original_em_clustering_np_matrix = np.load("NBA_EM_Original_results.npy")
pca_em_clustering_np_matrix = np.load("NBA_EM_D_Reduction_PCA_17d_results.npy")
ica_original_em_clustering_np_matrix = np.load("NBA_EM_D_Reduction_ICA_original_13d_results.npy")
rp_original_em_clustering_np_matrix = np.load("NBA_EM_D_Reduction_RP_origininal_18d_results.npy")
rp_normalized_em_clustering_np_matrix = np.load("NBA_EM_D_Reduction_RP_normalized_17d_results.npy")

all_em_clustering_np_matrix = {
'Normalized':    normalized_em_clustering_np_matrix,
'Original':    original_em_clustering_np_matrix,
'PCA':    pca_em_clustering_np_matrix,
'ICA_original':    ica_original_em_clustering_np_matrix,
# 'ICA_normalized':    ica_normalized_kmeans_clustering_np_matrix,
'RP_original':    rp_original_em_clustering_np_matrix,
'RP_normalized':    rp_normalized_em_clustering_np_matrix
}

colors = ['darkorchid', 'turquoise', 'darkorange', 'crimson', 'green',
          'dodgerblue', 'grey', 'greenyellow', 'navy', 'aqua',
          'brown', 'crimson', 'darkgoldenrod']

def plot_boxplot():
    columns = 3
    rows = 2
    step = 1 + num_features / (columns * rows)
    print 'step', step
    # fig, ax = plt.subplots(width, hight)
    start = 0
    end = step
    plt.figure(figsize=(16, 9))
    for i in range(rows):
        for j in range(columns):
            idx = 100*rows + 10*columns + i*columns+j+1
            plt.subplot(idx)
            plt.boxplot(player_stat_normalized[:, start:end], 0, '')
            plt.xticks(np.arange(step),players_stat_columns[start:end],rotation=70)
            plt.xticks(fontsize=6)
            start += step
            end += step
    plt.subplots_adjust(top=0.92, bottom =0.2, hspace = 0.4)
    plt.suptitle('Normalized Data', fontsize=16)
    plt.savefig('NormalizedData_BoxPlot.png')
    plt.close()
# plot_boxplot()

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
    # index of row = desired number of clusters - 5
    # cluster_results = original_kmeans_clustering_np_matrix[49]
    # cluster_results = normalized_kmeans_clustering_np_matrix[49]
    cluster_results = ica_original_kmeans_clustering_np_matrix[49]
    num_clusters=np.amax(cluster_results)+1
    print 'num_clusters', num_clusters
    for cluster in range(num_clusters):
        print cluster
        names_list = players_name[cluster_results == cluster]
        efficiency_list = players_dataframe.Efficiency.values[cluster_results == cluster]
        avg_shot_dist_list = list(players_dataframe.Avg_Shot_Dis_ft[cluster_results == cluster])
        result_list = [names_list[idx]
                       +'_'+str(int(efficiency_list[idx]))
                       +'_'
                       +str(int(avg_shot_dist_list[idx]))
                        for idx in range(len(names_list))]
        print result_list


# kmeans_plot_pca_3d()
# kmeans_plot_clustering_names()

def em_plot_clustering_names():
    # index of row = desired number of clusters - 5
    # cluster_results = original_em_clustering_np_matrix[49]
    # cluster_results = normalized_em_clustering_np_matrix[51]
    cluster_results = original_em_clustering_np_matrix[51]
    num_clusters=np.amax(cluster_results)+1
    print 'num_clusters', num_clusters
    for cluster in range(num_clusters):
        print cluster
        names_list = players_name[cluster_results == cluster]
        efficiency_list = players_dataframe.Efficiency.values[cluster_results == cluster]
        avg_shot_dist_list = list(players_dataframe.Avg_Shot_Dis_ft[cluster_results == cluster])
        result_list = [names_list[idx]
                       +'_'+str(int(efficiency_list[idx]))
                       +'_'
                       +str(int(avg_shot_dist_list[idx]))
                        for idx in range(len(names_list))]
        print result_list
# em_plot_clustering_names()

def append_label_to_file(n_clusters):
    row_idx = n_clusters - 5
    for k,v in all_kmeans_clustering_np_matrix.items():
        column_name = 'label_' + k
        labels = v[row_idx]
        players_dataframe[column_name] = labels
    players_dataframe.to_csv('nba_player_whole_stats_labels.csv')

def plot_kmenas_v_measure(n_clusters):
    row_idx = n_clusters - 5
    v_measure_array = []
    xvalues = []
    true_labels = pca_em_clustering_np_matrix[row_idx]
    for k,v in all_kmeans_clustering_np_matrix.items():
        xvalues.append(k)
        c = completeness_score(true_labels, v[row_idx])
        h = homogeneity_score(true_labels, v[row_idx])
        v_measure = 2 * c * h / (c + h)
        v_measure_array.append(v_measure)
    plt.figure(figsize=(8, 8))
    plt.plot(range(len(v_measure_array)), v_measure_array, "o-")
    plt.xticks(np.arange(len(v_measure_array)), xvalues)
    plt.yticks(np.linspace(0.6, 1.0, 9))
    plt.grid(True)
    plt.xlabel("\nK-means Clustering",fontsize=14)
    plt.ylabel("V-measure score",fontsize=14)
    plt.title("PCA EM clustering results are set as true lables",fontsize=14)
    plt.show()

def plot_em_v_measure(n_clusters):
    row_idx = n_clusters - 5
    v_measure_array = []
    xvalues = []
    true_labels = pca_em_clustering_np_matrix[row_idx]
    for k,v in all_em_clustering_np_matrix.items():
        xvalues.append(k)
        c = completeness_score(true_labels, v[row_idx])
        h = homogeneity_score(true_labels, v[row_idx])
        v_measure = 2 * c * h / (c + h)
        v_measure_array.append(v_measure)
    plt.figure(figsize=(8, 8))
    plt.plot(range(len(v_measure_array)), v_measure_array, "o-")
    plt.xticks(np.arange(len(v_measure_array)), xvalues)
    plt.ylim(0.6,1.0)
    plt.yticks(np.linspace(0.6, 1.0, 9))
    plt.grid(True)
    plt.xlabel("\nEM clustering",fontsize=14)
    plt.ylabel("V-measure score",fontsize=14)
    plt.title("PCA EM clustering results are set as true lables",fontsize=14)
    plt.show()

def plot_kmeans_em_v_meansrue(n_clusters):
    row_idx = n_clusters - 5
    v_measure_array = []
    xvalues = all_em_clustering_np_matrix.keys()
    for key in xvalues:
        kmeans_label = all_kmeans_clustering_np_matrix[key][row_idx]
        em_label = all_em_clustering_np_matrix[key][row_idx]
        c = completeness_score(kmeans_label, em_label)
        h = homogeneity_score(kmeans_label, em_label)
        v_measure = 2 * c * h / (c + h)
        v_measure_array.append(v_measure)
    plt.figure(figsize=(8, 8))
    plt.plot(range(len(v_measure_array)), v_measure_array, "o-")
    plt.xticks(np.arange(len(v_measure_array)), xvalues)
    plt.yticks(np.linspace(0.6, 1.0, 9))
    plt.grid(True)
    plt.xlabel("\nDatasets",fontsize=14)
    plt.ylabel("V-measure score",fontsize=14)
    plt.title("Comparison of clustering consistency \nbetween EM and Kmeans on the same dataset",fontsize=14)
    plt.show()

def append_em_pca_label_to_file(n_clusters):



# plot_kmenas_v_measure(54)
plot_em_v_measure(54)
# plot_kmeans_em_v_meansrue(54)
# append_label_to_file(54)