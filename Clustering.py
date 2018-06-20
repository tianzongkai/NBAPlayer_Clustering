from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
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
players_stat = players_dataframe.values[:,3:]
print players_stat.shape


def kmeans():
    initialization = 'k-means++'  # 'k-means++'
    min = 5
    max = 40
    step = 5
    k_range = np.array(range(min, max))
    varing_k_cluster_labels = []
    filename = ('k_means_%s_%s') % (initialization, time.strftime("%m%d%H%M%S"))
    silhouette_scores = np.array([])
    sse_array = np.array([])
    for i in k_range:
        print i
        kmeans = KMeans(init=initialization, n_clusters=i, n_init=30).fit(players_stat)
        cluster_labels = kmeans.fit_predict(players_stat)
        varing_k_cluster_labels.append(cluster_labels)
        silhouette = metrics.silhouette_score(players_stat, cluster_labels)
        sse = kmeans.inertia_ / num_players
        silhouette_scores = np.append(silhouette_scores, silhouette)
        sse_array = np.append(sse_array, sse)
    np.save(filename, varing_k_cluster_labels)
    plt.close()
    # print "silhouette coefficient of original labels %0.03f" % metrics.silhouette_score(players_shoot_stat, labels)

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
    plt.yticks(np.linspace(0.17, 0.25, 9))
    plt.ylabel('Silhouette')
    plt.xlabel('# of clusters')
    # plt.legend()
    plt.suptitle(("NBA players k-means clustering\nDeciding value of k with %s initilization") % initialization)
    plt.savefig(("%s.png") % filename)
    # plt.show()

# for _ in range(5):
#     kmeans()
colors = ['darkorchid', 'turquoise', 'darkorange', 'crimson', 'green',
          'dodgerblue', 'grey', 'greenyellow', 'navy', 'aqua',
          'brown', 'crimson', 'darkgoldenrod']

def kmeans_original_data_clusters_plot():
    cluster_results = np.load("k_means_k-means++_11clusters.npy") # a matrix of 35 x 452, num of clusters varing from 5 to 39
    cluster_results = cluster_results[6] # pick the row of 11 clusters
    num_clusters=np.amax(cluster_results)+1
    print num_clusters
    for color, cluster_label in zip(colors, range(num_clusters)):
        print cluster_label
        plt.subplot(121)
        plt.scatter(players_stat[cluster_results == cluster_label, 5],
                    players_stat[cluster_results == cluster_label, 9], color=color,
                    label=cluster_label, marker='+')
        plt.xlabel('Above the break-3 Usage In the paint (Non-RA) Usage', fontsize=7)
        plt.xticks(fontsize=7)
        plt.yticks(fontsize=7)
        plt.suptitle("NBA Players divided into 9 groups \nbased on shooting distance & zone")
        plt.subplot(122)
        plt.scatter(players_stat[cluster_results == cluster_label, 8],
                    players_stat[cluster_results == cluster_label, 16], color=color,
                    label=cluster_label, marker='+')
        plt.xlabel('Mid-range Usage vs Restricted Area Usage', fontsize=7)
        plt.xticks(fontsize=7)
        plt.yticks(fontsize=7)
    # plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.show()

kmeans_original_data_clusters_plot()

