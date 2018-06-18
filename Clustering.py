from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics


players_data = pd.read_csv("nba_player_shotting.csv")
print players_data.columns
players_shoot_stat = players_data.values[:,4:]
players_name_team = players_data.values[:,1:3]
min = 5
max = 40
step = 5
k_range = np.array(range(min,max))
def kmeans():
    silhouette_scores = np.array([])
    sse_array = np.array([])
    completeness_array = np.array([])
    for i in k_range:
        print i
        kmeans = KMeans(init='k-means++', n_clusters=i, n_init=10).fit(players_shoot_stat)
        cluster_labels = kmeans.fit_predict(players_shoot_stat)
        silhouette = metrics.silhouette_score(players_shoot_stat, cluster_labels)
        sse = kmeans.inertia_ / players_name_team.shape[0]
        silhouette_scores = np.append(silhouette_scores, silhouette)
        sse_array = np.append(sse_array, sse)

    # print "silhouette coefficient of original labels %0.03f" % metrics.silhouette_score(players_shoot_stat, labels)

    plt.subplot(121)
    plt.plot(k_range, sse_array, "ro-", label='Avg Sum of Squared Error')
    plt.grid(True)
    plt.xticks(range(min, max, step))
    plt.ylabel('Avg SSE')
    plt.xlabel('# of clusters')
    plt.legend()
    # plt.title("NBA Players, k-means")

    plt.subplot(122)
    plt.plot(k_range, silhouette_scores, "go-", label="Silhouette Coefficient")
    plt.grid(True)
    plt.xticks(range(min, max, step))
    plt.ylabel('Silhouette')
    plt.xlabel('# of clusters')
    plt.legend()
    plt.suptitle("NBA players k-means clustering based on shooting zone & distance statistics\nSelecting number of clusters")

    plt.show()

kmeans()
colors = ['darkorchid', 'turquoise', 'darkorange', 'crimson', 'green', 'dodgerblue', 'grey', 'greenyellow','navy']
def kmeans_original_data_9clusters_plot():
    num_clusters=9
    kmeans = KMeans(init='k-means++', n_clusters=num_clusters, n_init=10).fit(players_shoot_stat)
    cluster_labels = kmeans.fit_predict(players_shoot_stat)
    for color, target_name in zip(colors, range(num_clusters)):
        plt.subplot(121)
        plt.scatter(players_shoot_stat[cluster_labels == target_name, 0],
                    players_shoot_stat[cluster_labels == target_name, 9], color=color,
                    label=target_name, marker='+')
        plt.xlabel('Above the break-3 Usage In the paint (Non-RA) Usage', fontsize=7)
        plt.xticks(fontsize=7)
        plt.yticks(fontsize=7)
        plt.suptitle("NBA Players divided into 9 groups \nbased on shooting distance & zone")
        plt.subplot(122)
        plt.scatter(players_shoot_stat[cluster_labels == target_name, 1],
                    players_shoot_stat[cluster_labels == target_name, 3], color=color,
                    label=target_name, marker='+')
        plt.xlabel('Mid-range Usage vs Restricted Area Usage', fontsize=7)
        plt.xticks(fontsize=7)
        plt.yticks(fontsize=7)
    # plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.show()

kmeans_original_data_9clusters_plot()

