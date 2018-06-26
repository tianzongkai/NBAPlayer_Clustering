from sklearn.cluster import KMeans
import numpy as np
from numpy import linalg as la
from matplotlib import cm
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics, manifold, preprocessing
from sklearn.decomposition import PCA, FastICA
from scipy import spatial as sp
from scipy import stats as sta
from sklearn import mixture, random_projection
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
print 'shape:', players_dataframe.shape #452x124, 411x124 after remove <15 games
players_dataframe.fillna(value=0.0, inplace=True)
for column in players_dataframe.filter(regex='Pct').columns:
    players_dataframe[column] = players_dataframe[column].apply(lambda x:x*100)

# print players_dataframe.And_One
# missing_values_count = players_dataframe.isnull().sum()
# print type(missing_values_count)
# print missing_values_count
# print len(players_dataframe.loc[players_dataframe.And_One.isnull()].Player)


# print players_data.columns
players_stat = players_dataframe.values[:,3:] # np array
num_features = players_stat.shape[1]

min_max_scaler = preprocessing.MinMaxScaler()
players_stat_normalized = min_max_scaler.fit_transform(players_stat)
players_stat_normalized_df = pd.DataFrame(players_stat_normalized)

def correlation_egienvalue():
    cov = np.cov(players_stat_normalized, rowvar=0)
    corr = np.corrcoef(players_stat_normalized,rowvar=0)
    eigvalues, eigenvectors = la.eig(cov)
    abs_cov = np.absolute(cov)
    abs_corr = np.absolute(corr)
    #cov = np.dot(non_label_dataset.T, non_label_dataset)
    # print abs_corr
    # print eigvalues
    plt.figure(figsize=(18, 9))
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
    plt.savefig("original_Correlation_Eigenvalues.png")
    plt.close()
    # plt.legend()
    # plt.show()

def pca_varing_k():
    print 'pca_varing_k'
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
    plt.figure(figsize=(16, 9))
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
    plt.suptitle("PCA - Deciding # of components for clustering")
    plt.savefig('PCA_deciding_num_components.png')
    plt.close()

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

def ica_varing_components(data, type):
    n_components_max = 56 # num_features+1
    n_components = np.arange(2, n_components_max, 1)
    kurtosis_matrix = np.array([])
    for n in n_components:
        print n
        ica = FastICA(n_components=n,whiten=True, algorithm='deflation', max_iter=100)
        ica.fit(data)
        transformed_data = ica.transform(data)
        #kurtosis = sta.kurtosis(transformed_data, axis=None)
        # print sta.kurtosis(transformed_data, axis=None)
        kurtosis = sta.kurtosis(transformed_data) # for Normal distribution, kurtosis = 0 by this algorithm
        kurtosis_matrix = np.append(kurtosis_matrix, np.average(kurtosis))
    plt.figure(figsize=(16, 9))
    # print kurtosis_matrix
    plt.plot(n_components, kurtosis_matrix, "o-", label="kurtosis")
    plt.grid(True)
    plt.xlim(np.amin(n_components), np.amax(n_components))
    # plt.ylim(1e-17,1e-1)
    # plt.yscale('log')
    plt.xticks(range(np.amin(n_components), n_components_max+3, 3))
    plt.ylabel('kurtosis')
    plt.xlabel('# of components')
    plt.legend()
    plt.title(("NBA Player Stats %s ICA\n(Kurtosis of Normal Distribution is 0 in this algorithm)") % (type))
    plt.savefig(("original_ICA_kurtosis_2_to_%d_components.png") % n_components_max)
    plt.close()

def run_ica():
    print 'run_ica'
    datas = [players_stat]
    types = ['Un_normalized']
    for data, type in zip(datas, types):
        ica_varing_components(data, type)

def ica_original_25_components():
    filename = ("nba_original_ica_transformed_25d_matrix.npy")
    ica = FastICA(n_components=37, algorithm='deflation', max_iter=100)
    ica.fit(players_stat)
    transformed_data = ica.transform(players_stat)
    np.save(filename,transformed_data)

def rp(data, type):
    filename_template = "nba_{type}_rp_transformed_{dimension}d_matrix.npy"
    iteration = 50
    n_components_min = 2
    n_components_max = 20
    n_components = np.arange(n_components_min, n_components_max, 1)
    x_value = np.repeat(n_components, iteration)
    distortion_array = np.array([])
    least_distortion = float('Inf')
    least_distortion_dimension = 0
    best_transformed_data = np.array([])
    origin_dist_matrix = np.asarray([[la.norm(u - v) for v in data] for u in data])
    def calculate_distortion(transformed_data):
        size = transformed_data.shape[0]
        max_distortion = float('-inf')
        for u in range(size):
            for v in range(size):
                if v < u:
                    origin_dist = origin_dist_matrix[u,v]
                    transformed_dist = la.norm(transformed_data[u] - transformed_data[v])
                    distortion = (transformed_dist / origin_dist) ** 2
                    if distortion > max_distortion: max_distortion = distortion
        return max_distortion

    for n in n_components:
        print n
        for i in range(iteration):
            rp = random_projection.GaussianRandomProjection(n_components=n,eps=0.1)
            transformed_data = rp.fit_transform(data)
            distortion = calculate_distortion(transformed_data)
            distortion_array = np.append(distortion_array, distortion)
            if distortion < least_distortion:
                least_distortion = distortion
                best_transformed_data = transformed_data
                least_distortion_dimension = n
    # print "# of components: %r" % best_transformed_data.shape[1]
    # print "least_f_norm_percent_change is %.2f%%" % least_f_norm_percent_change
    filename = filename_template.format(type = type, dimension=str(least_distortion_dimension))
    np.save(filename,best_transformed_data)
    plt.figure(figsize=(16, 9))
    plt.scatter(x_value, distortion_array, marker='+')
    # plt.xlim(n_components_min, n_components_max)
    plt.xticks(np.arange(n_components_min-1,n_components_max+1,1))
    # plt.yscale('log')
    plt.grid(True)
    plt.xlabel("# of components")
    plt.ylabel("Distortion")
    note = "Least distortion: %.2f" % (least_distortion)
    notex, notey = best_transformed_data.shape[1], least_distortion
    plt.title("NBA Players Stats, Randomized Projects %s\n %r iterations for each # of components" % (type, iteration))
    plt.annotate(note, xy=(notex ,notey), xytext=(notex + 0.2,notey + 0.2), wrap=True,
        arrowprops=dict(facecolor='black', shrink=0.005))
    plt.savefig(("random_projection_distortion_%s.png") % type)
    plt.close()

def run_rp():
    print 'run_rp'
    datas = [players_stat, players_stat_normalized]
    types = ['Normalized']
    # types = ['Un_normalized']
    for data, type in zip(datas, types):
        rp(data, type)


# correlation_egienvalue()
# pca_varing_k()
# pca_17_components()
# pca_3_components()
# run_ica()
# ica_original_25_components()
# run_rp()
#
