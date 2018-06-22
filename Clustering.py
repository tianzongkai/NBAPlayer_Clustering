from sklearn.cluster import KMeans
import numpy as np
from numpy import linalg as la
from matplotlib import cm
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn import preprocessing
from sklearn.decomposition import PCA, FastICA
import time


# players_data = pd.read_csv("nba_player_whole_stats_edited_column_name.csv")
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
players_stat = players_dataframe.values[:,3:] # np array
num_features, num_players = players_stat.shape

min_max_scaler = preprocessing.MinMaxScaler()
np_scaled = min_max_scaler.fit_transform(players_stat)
players_stat_normalized = pd.DataFrame(np_scaled)
players_name = players_dataframe.values[:,1] # np array


print num_features, num_players


def kmeans(type):
    if type == 'Original':
        data = players_stat
    else:
        data = players_stat_normalized
    initialization = 'k-means++'  # 'k-means++'
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
        kmeans = KMeans(init=initialization, n_clusters=i, n_init=30).fit(data)
        cluster_labels = kmeans.fit_predict(data)
        varing_k_cluster_labels.append(cluster_labels)
        silhouette = metrics.silhouette_score(data, cluster_labels)
        sse = kmeans.inertia_ / num_players
        silhouette_scores = np.append(silhouette_scores, silhouette)
        sse_array = np.append(sse_array, sse)
    np.save(filename, varing_k_cluster_labels)
    plt.close()
    # print "silhouette coefficient of original labels %0.03f" % metrics.silhouette_score(players_shoot_stat, labels)
    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    plt.plot(k_range, sse_array, "ro-", label='Avg Sum of Squared Error')
    plt.grid(True)
    plt.xticks(range(min, max, step))
    # plt.yticks(np.linspace(0.14, 0.23, 10))
    plt.ylabel('Avg Sum of Squared Error')
    plt.xlabel('# of clusters')
    # plt.legend()
    # plt.title("NBA Players, k-means")

    plt.subplot(122)
    plt.plot(k_range, silhouette_scores, "go-", label="Silhouette Coefficient")
    plt.grid(True)
    plt.xticks(range(min, max, step))
    if type == 'Original':
        plt.yticks(np.linspace(0.15, 0.25, 11))
    plt.ylabel('Silhouette')
    plt.xlabel('# of clusters')
    # plt.legend()
    plt.suptitle(("NBA players %s k-means++ clustering\nDeciding value of k with %s initilization") % (type, initialization))
    plt.savefig(("%s.png") % filename)
    # plt.show()

for _ in range(5):
    # type = 'Original'
    type = 'Normalized'
    kmeans(type)

# kmeans_original_data_clusters_plot()


