from sklearn.cluster import KMeans
import numpy as np
from numpy import linalg as la
from matplotlib import cm
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics, mixture
from sklearn import preprocessing
from sklearn.decomposition import PCA, FastICA
import time


players_dataframe = pd.read_csv("nba_player_whole_stats.csv")
players_dataframe.drop(players_dataframe[players_dataframe.Games < 15].index, inplace=True)
players_dataframe.drop(columns='Total_Plus_Minus', inplace=True)


print 'shape:', players_dataframe.shape
players_dataframe.fillna(value=0.0, inplace=True)

# Apply operations on each row or column with the lambda function
# x is a variable representing each row or column
for column in players_dataframe.filter(regex='Pct').columns:
    players_dataframe[column] = players_dataframe[column].apply(lambda x:x*100)

players_stat = players_dataframe.values[:,3:] # (411, 119) after remove <15 games
num_features, num_players = players_stat.shape

min_max_scaler = preprocessing.MinMaxScaler()
player_stat_normalized = min_max_scaler.fit_transform(players_stat)
players_stat_normalized_df = pd.DataFrame(player_stat_normalized)
players_name = players_dataframe.values[:,1] # np array
print num_features, num_players

# Load previously saved transformed data; variable is numpy array
pca_17d_transformed_data = np.load( "nba_pca_transformed_17d_matrix.npy")
pca_3d_transformed_data = np.load( "nba_pca_transformed_3d_matrix.npy")
ica_original_25d_transformed_data = np.load("nba_original_ica_transformed_25d_matrix.npy")
rp_original_18d_transformed_data = np.load( "nba_Un_normalized_rp_transformed_18d_matrix.npy")
rp_normalized_17d_transformed_data = np.load("nba_Normalized_rp_transformed_17d_matrix.npy")

original_dataset_dict = {
    'Original': players_stat,
    'Normalized': player_stat_normalized
}


# put all transformed data in a dictionary
d_reduction_dataset_dict = {
    'PCA_17d': pca_17d_transformed_data,
#     'pca_3d': pca_3d_transformed_data,
    'ICA_original_25d': ica_original_25d_transformed_data,
    'RP_origininal_18d': rp_original_18d_transformed_data,
    'RP_normalized_17d': rp_normalized_17d_transformed_data
    }

def kmeans(type):
    # run kmeans on original 119 features
    if type == 'Original':
        data = players_stat
    else:
        data = player_stat_normalized
        
    # initalizing centering with k-means++ algorithm to ensure initial centers are well apart
    initialization = 'k-means++'
    min = 5
    max = 81
    step = 5
    k_range = np.array(range(min, max))
    varing_k_cluster_labels = []
    filename = ('%s_k_means_%s_%s') % (type, initialization, time.strftime("%m%d%H%M%S"))
    silhouette_scores = np.array([])
    sse_array = np.array([])
    for i in k_range:
        print i
        kmeans = KMeans(init=initialization, n_clusters=i, n_init=50).fit(data)
        cluster_labels = kmeans.fit_predict(data)
        varing_k_cluster_labels.append(cluster_labels)
        
        silhouette = metrics.silhouette_score(data, cluster_labels)        
        silhouette_scores = np.append(silhouette_scores, silhouette)
        
        # sum of squared error
        sse = kmeans.inertia_ / num_players
        sse_array = np.append(sse_array, sse)
    np.save(filename, varing_k_cluster_labels)
    plt.close()
    # print "silhouette coefficient of original labels %0.03f" % metrics.silhouette_score(players_shoot_stat, labels)
    plt.figure(figsize=(12, 6))

    plt.subplot(121)
    plt.plot(k_range, sse_array, "ro-", label='Avg Sum of Squared Error')
    plt.grid(True)
    plt.xticks(range(min, max, step))
    plt.ylabel('Avg Sum of Squared Error')
    plt.xlabel('# of clusters')
    # plt.legend()
    # plt.title("NBA Players, k-means")

    plt.subplot(122)
    if type == 'Original':
        min_s = 0.11 # np.amin(silhouette_scores)-0.01
        max_s = 0.23 # np.amax(silhouette_scores)+0.01
    else:
        min_s = 0.04 # np.amin(silhouette_scores)-0.01
        max_s = 0.16 # np.amax(silhouette_scores)+0.01
    y_sapce = int((max_s-min_s)/0.01)+1
    print min_s, max_s, y_sapce
    plt.plot(k_range, silhouette_scores, "go-", label="Silhouette Coefficient")
    plt.grid(True)
    plt.xticks(range(min, max, step))
    # plt.yticks(np.linspace(min_s, max_s, y_sapce))
    plt.ylabel('Silhouette')
    plt.xlabel('# of clusters')
    # plt.legend()
    plt.suptitle(("NBA players %s k-means++ clustering\nDeciding value of k with %s initilization\n50 runs on each k") % (type, initialization))
    plt.savefig(("%s.png") % filename)
    # plt.show()

def run_kmeans():
    for type in ['Original', 'Normalized']:
            kmeans(type)
# run_kmeans()

def reduced_kmeans():
    # run kmeans on d-reducted data
    filename_template = "NBA_D_Reduction_{type}_kmeans_plus_results{extension}"
    min = 5
    max = 81
    step = 5
    k_range = np.array(range(min, max))
    for k, v in d_reduction_dataset_dict.items():
        varing_k_cluster_labels = []
        silhouette_scores = np.array([])
        sse_array = np.array([])
        print k
        for i in k_range:
            if i % 10 == 0: print i
            kmeans = KMeans(init='k-means++', n_clusters=i, n_init=50).fit(v)
            cluster_labels = kmeans.fit_predict(v)
            varing_k_cluster_labels.append(cluster_labels)
            silhouette = metrics.silhouette_score(v, cluster_labels)
            sse = kmeans.inertia_ / num_players
            silhouette_scores = np.append(silhouette_scores, silhouette)
            sse_array = np.append(sse_array, sse)

        filename = filename_template.format(type=k, extension='.npy')
        np.save(filename, varing_k_cluster_labels)

        plt.figure(figsize=(12, 5))
        plt.subplot(121)
        plt.plot(k_range, sse_array, "ro-", label='Avg Sum of Squared Error')
        plt.grid(True)
        plt.xticks(range(min, max, step))
        plt.ylabel('Avg Sum of Squared Error')
        plt.xlabel('# of clusters')

        plt.subplot(122)
        min_s = 0.01  # np.amin(silhouette_scores)-0.01
        max_s = 0.20  # np.amax(silhouette_scores)+0.01
        y_sapce = int((max_s - min_s) / 0.01) + 1
        # print min_s, max_s, y_sapce
        plt.plot(k_range, silhouette_scores, "go-", label="Silhouette Coefficient")
        plt.grid(True)
        plt.xticks(range(min, max, step))
        # plt.yticks(np.linspace(min_s, max_s, y_sapce))
        plt.ylabel('Silhouette')
        plt.xlabel('# of clusters')
        plt.suptitle(
            ("NBA players Dimensionality Reduction k-means++ clustering\nDeciding value of k - %s")
            % (k))
        filename = filename_template.format(type=k, extension='.png')
        plt.savefig(filename)
        plt.close()

# reduced_kmeans()

def em():
    # Run Gaussian Mixture model on 119 features
    filename_template = "NBA_EM_{type}_results{extension}"
    min = 5
    max = 81
    step = 5
    k_range = np.array(range(min, max))
    for k, v in original_dataset_dict.items():
        varing_k_cluster_labels = []
        silhouette_scores = np.array([])
        print k
        for i in k_range:
            if i % 10 == 0: print i
            em = mixture.GaussianMixture(n_components=i,n_init=10, covariance_type='full')
            em.fit(v)
            cluster_labels = em.predict(v)
            varing_k_cluster_labels.append(cluster_labels)
            silhouette = metrics.silhouette_score(v, cluster_labels)
            silhouette_scores = np.append(silhouette_scores, silhouette)

        filename = filename_template.format(type=k, extension='.npy')
        np.save(filename, varing_k_cluster_labels)

        plt.figure(figsize=(6, 5))
        # min_s = 0.06  # np.amin(silhouette_scores)-0.01
        # max_s = 0.25  # np.amax(silhouette_scores)+0.01
        # y_sapce = int((max_s - min_s) / 0.01) + 1
        # print min_s, max_s, y_sapce
        plt.plot(k_range, silhouette_scores, "go-", label="Silhouette Coefficient")
        plt.grid(True)
        plt.xticks(range(min, max, step))
        # plt.yticks(np.linspace(min_s, max_s, y_sapce))
        plt.ylabel('Silhouette')
        plt.xlabel('# of clusters')
        plt.suptitle(
            ("NBA players EM clustering\nDeciding value of k\n%s")
            % (k))
        filename = filename_template.format(type=k, extension='.png')
        plt.savefig(filename)
        plt.close()

def reduced_em():
     # Run Gaussian Mixture model on d-reduced data
    filename_template = "NBA_EM_D_Reduction_{type}_results{extension}"
    min = 5
    max = 81
    step = 5
    k_range = np.array(range(min, max))
    for k, v in d_reduction_dataset_dict.items():
        varing_k_cluster_labels = []
        silhouette_scores = np.array([])
        print k
        for i in k_range:
            if i % 10 == 0: print i
            em = mixture.GaussianMixture(n_components=i, covariance_type='full')
            em.fit(v)
            cluster_labels = em.predict(v)
            varing_k_cluster_labels.append(cluster_labels)
            silhouette = metrics.silhouette_score(v, cluster_labels)
            silhouette_scores = np.append(silhouette_scores, silhouette)

        filename = filename_template.format(type=k, extension='.npy')
        np.save(filename, varing_k_cluster_labels)

        plt.figure(figsize=(6, 5))
        min_s = 0.05  # np.amin(silhouette_scores)-0.01
        max_s = 0.25  # np.amax(silhouette_scores)+0.01
        y_sapce = int((max_s - min_s) / 0.01) + 1
        # print min_s, max_s, y_sapce
        plt.plot(k_range, silhouette_scores, "go-", label="Silhouette Coefficient")
        plt.grid(True)
        plt.xticks(range(min, max, step))
        plt.yticks(np.linspace(min_s, max_s, y_sapce))
        plt.ylabel('Silhouette')
        plt.xlabel('# of clusters')
        plt.suptitle(
            ("NBA players Dimensionality Reduction em clustering\n%s")
            % (k))
        filename = filename_template.format(type=k, extension='.png')
        plt.savefig(filename)
        plt.close()

# em()
# reduced_em()
