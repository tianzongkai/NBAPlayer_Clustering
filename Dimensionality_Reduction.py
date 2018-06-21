from sklearn.cluster import KMeans
import numpy as np
from numpy import linalg as la
from matplotlib import cm
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn import preprocessing
from sklearn.decomposition import PCA, FastICA
from scipy import spatial as sp
from scipy import stats as sta
from sklearn import mixture, random_projection
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
players_stat = players_dataframe.values[:,3:] # np array
num_features = players_stat.shape[1]

min_max_scaler = preprocessing.MinMaxScaler()
np_scaled = min_max_scaler.fit_transform(players_stat)
players_stat_normalized = pd.DataFrame(np_scaled)

def correlation_egienvalue():
    cov = np.cov(players_stat_normalized, rowvar=0)
    corr = np.corrcoef(players_stat_normalized,rowvar=0)
    eigvalues, eigenvectors = la.eig(cov)
    abs_cov = np.absolute(cov)
    abs_corr = np.absolute(corr)
    #cov = np.dot(non_label_dataset.T, non_label_dataset)
    print abs_corr
    print eigvalues

    plt.subplot(131)
    cmap = cm.get_cmap()
    cax = plt.imshow(abs_corr, interpolation="nearest", cmap=cmap)
    #plt.grid(True)
    plt.title('Feature Correlation')
    # labels = range(num_features)
    # plt.xlabel(labels, fontsize=9)
    # plt.ylabel(labels, fontsize=9)
    plt.xticks(range(0, num_features, 10))
    plt.yticks(range(0, num_features, 10))
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    plt.colorbar(cax, ticks=np.arange(0.1,1.2,0.1))

    plt.subplot(132)
    plt.plot(range(1, num_features+1)[:11], eigvalues[:11], "o-", label="eigenvalues")
    plt.grid(True)
    plt.xticks(range(1, 12))
    plt.ylabel('Eigenvalue')
    plt.xlabel('Index of eigenvalue')
    # plt.legend()
    plt.title("First 11 eigenvalues")

    plt.subplot(133)
    plt.plot(range(1, num_features+1)[11:], eigvalues[11:], "o-", label="eigenvalues")
    plt.grid(True)
    plt.xticks(range(12, num_features+2, 10))
    plt.ylabel('Eigenvalue')
    plt.xlabel('Index of eigenvalue')
    plt.title("Rest eigenvalues")
    # plt.legend()
    plt.show()

def pca_varing_k():
    n_components = np.arange(2,20,1)
    retained_variance_vs_components = np.array([])
    distance_vs_components = np.array([])
    print n_components
    for n in n_components:
        pca = PCA(n_components=n, svd_solver='full')
        pca.fit(players_stat_normalized)
        sum_variance_ration = np.sum(pca.explained_variance_ratio_)
        retained_variance_vs_components = np.append(retained_variance_vs_components, sum_variance_ration)
        transformed_data = pca.transform(players_stat_normalized)
        reconstructed_data = pca.inverse_transform(transformed_data)
        #dist_matrix = sp.distance.cdist(non_label_dataset, reconstructed_data)
        #dist_matrix = la.norm(non_label_dataset - reconstructed_data)
        dist_matrix = 100 * abs((la.norm(players_stat_normalized) - la.norm(reconstructed_data)))/ la.norm(players_stat_normalized)
        sum_dist = np.sum(dist_matrix)
        distance_vs_components = np.append(distance_vs_components, sum_dist)
    # print retained_variance_vs_components
    # print distance_vs_components

    plt.subplot(121)
    plt.plot(n_components, retained_variance_vs_components, "o-", label="retained variance ratio")
    plt.grid(True)
    # plt.xlim(2, 9)
    #plt.ylim(0, 0.7)
    plt.xticks(n_components)
    plt.ylabel('retained variance ratio')
    plt.xlabel('# of components')
    plt.legend()

    plt.subplot(122)
    plt.plot(n_components, distance_vs_components, "go-", label="frobenius norm % change \nof projected-back data")
    plt.grid(True)
    # plt.xlim(2, 9)
    #plt.ylim(0.06, 0.14)
    plt.xticks(n_components)
    plt.ylabel('frobenius norm % change')
    plt.xlabel('# of components')
    plt.legend()
    plt.suptitle("PCA on normalized data\nDeciding # of components for clustering")
    plt.show()

def pca_17_components():
    filename = "nba_pca_transformed_17d_matrix.npy"
    pca = PCA(n_components=17, svd_solver='full')
    pca.fit(players_stat_normalized)
    transformed_data = pca.transform(players_stat_normalized)
    np.save(filename, transformed_data)

def pca_3_components():
    filename = "nba_pca_transformed_3d_matrix.npy"
    pca = PCA(n_components=3, svd_solver='full')
    pca.fit(players_stat_normalized)
    transformed_data = pca.transform(players_stat_normalized)
    np.save(filename, transformed_data)

def ica_varing_components():
    n_components = np.arange(2, num_features+1, 1)
    kurtosis_matrix = np.array([])
    for n in n_components:
        print n
        ica = FastICA(n_components=n,algorithm='deflation', max_iter=100)
        ica.fit(players_stat)
        transformed_data = ica.transform(players_stat)
        #kurtosis = sta.kurtosis(transformed_data, axis=None)
        # print sta.kurtosis(transformed_data, axis=None)
        kurtosis = sta.kurtosis(transformed_data) # for Normal distribution, kurtosis = 0 by this algorithm
        kurtosis_matrix = np.append(kurtosis_matrix, np.average(kurtosis))

    # print kurtosis_matrix
    plt.plot(n_components, kurtosis_matrix, "o-", label="kurtosis")
    plt.grid(True)
    plt.xlim(np.amin(n_components), np.amax(n_components))
    # plt.ylim(1e-17,1e-1)
    # plt.yscale('log')
    plt.xticks(range(2, num_features+1, 10))
    plt.ylabel('kurtosis')
    plt.xlabel('# of components')
    plt.legend()
    plt.title("NBA Player Stats ICA\nKurtosis of Normal Distribution is 0 in this algorithm")
    plt.show()

def rp():
    filename = "nba_rp_transformed_matrix.npy"
    iteration = 100
    n_components = np.arange(2, 16, 1)
    x_value = np.repeat(n_components, iteration)
    f_noram_array = np.array([])
    least_f_norm_percent_change = float('Inf')
    best_transformed_data = np.array([])
    for n in n_components:
        for i in range(iteration):
            rp = random_projection.GaussianRandomProjection(n_components=n,eps=0.1)
            transformed_data = rp.fit_transform(players_stat)
            reconstructed_data = np.dot(transformed_data, rp.components_)
            f_norm_percent_change = 100 * abs(
                (la.norm(players_stat) - la.norm(reconstructed_data)))/ la.norm(players_stat)
            f_noram_array = np.append(f_noram_array, f_norm_percent_change)
            if f_norm_percent_change < least_f_norm_percent_change:
                least_f_norm_percent_change = f_norm_percent_change
                best_transformed_data = transformed_data
    print "# of components: %r" % best_transformed_data.shape[1]
    print "least_f_norm_percent_change is %.2f%%" % least_f_norm_percent_change
    #np.save(filename,best_transformed_data)
    plt.scatter(x_value, f_noram_array, marker='+')
    plt.xlim(np.amin(n_components)-1, np.amax(n_components)+1)
    plt.xticks(np.arange(np.amin(n_components),np.amax(n_components),1))
    plt.yscale('log')
    plt.grid(True)
    plt.xlabel("# of components")
    plt.ylabel("Frobenius Norm Percentage Change %")
    note = "Best result %.2f%% f-norm change" % (least_f_norm_percent_change)
    notex, notey = best_transformed_data.shape[1], least_f_norm_percent_change
    plt.title("NBA Players Stats, Randomized Projects\n %r iterations for each # of components" % iteration)
    plt.annotate(note, xy=(notex ,notey), xytext=(notex + 0.2,notey + 0.2), wrap=True,
        arrowprops=dict(facecolor='black', shrink=0.005))
    # plt.savefig("C:\\Users\\Zongkai\\Google Drive\\GaTech\\2017Fall_MachineLearning\\Assignment3\\pics\\2\\Ablone_RP.jpg")
    plt.show()



# correlation_egienvalue()
# pca_varing_k()
# ica_varing_components()
rp()