import pandas as pd
import os, pdb, sys
import numpy as np
from datetime import datetime
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import time
import geoplot as gplt
import copy
import umap.umap_ as umap
import pylab
import seaborn as sns
import umap
from sklearn.preprocessing import StandardScaler
import pickle
import matplotlib.cm as cm
from mpl_toolkits import mplot3d
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import dendrogram
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from umap import UMAP
from sklearn.cluster import AgglomerativeClustering
import math
from sklearn.mixture import GaussianMixture


def data_process_data(data):
    data['starttime'] = data['starttime'].astype('datetime64')
    data['start_date_hr'] = data['starttime'].dt.floor('H')
    data['start_date'] = data['start_date_hr'].dt.date
    data['start_hr'] = data['start_date_hr'].dt.hour
    data['stoptime'] = data['stoptime'].astype('datetime64')
    data['stop_date_hr'] = data['stoptime'].dt.floor('H')
    data['stop_date'] = data['stop_date_hr'].dt.date
    data['stop_hr'] = data['stop_date_hr'].dt.hour
    data['start_weekday'] = data['starttime'].apply(lambda x: x.isoweekday())
    data['stop_weekday'] = data['stoptime'].apply(lambda x: x.isoweekday())
    return data


def summary_ride(data, key):
    mask = data[data[f'start {key}'] == data[f'end {key}']].index
    data = data.drop(index=mask)
    # summarize rides by starting/stop station and hr
    ride_start = data[['start_date_hr', f'start {key}']].groupby(
        ['start_date_hr', f'start {key}']).size().to_frame(name='start_count').reset_index()
    ride_end = data[['stop_date_hr', f'end {key}']].groupby(
        ['stop_date_hr', f'end {key}']).size().to_frame(name='end_count').reset_index()
    # calculate net rides by station at different hr
    ride = pd.merge(ride_start, ride_end, left_on=['start_date_hr', f'start {key}'],
                    right_on=['stop_date_hr', f'end {key}'], how='outer')
    ride['start_count'].fillna(0, inplace=True)
    ride['end_count'].fillna(0, inplace=True)
    ride['net_checkout'] = ride['start_count'] - ride['end_count']
    ride['date_hour'] = ride['start_date_hr'].fillna(ride['stop_date_hr'])
    ride['date'] = ride['date_hour'].dt.date
    ride['hour'] = ride['date_hour'].dt.hour

    ride[key] = ride[f'start {key}'].fillna(ride[f'end {key}'])
    ride = ride.drop(columns=['start_date_hr', f'start {key}', 'stop_date_hr', f'end {key}'])
    return ride


def summarize_station(data):
    # summarize station info
    col_names = ['station id', 'station name', 'station latitude', 'station longitude']

    station_start = data[['start station id', 'start station name',
                          'start station latitude', 'start station longitude']]
    station_start.columns = col_names
    station_end = data[['end station id', 'end station name',
                        'end station latitude', 'end station longitude']]
    station_end.columns = col_names
    data_station = station_start.append(station_end).drop_duplicates().reset_index(drop=True)
    return data_station


def area_concat(key, df):
    df_area = df
    df_area.columns = [key + ' ' + x for x in df_area.columns]
    return df_area


def summarize_rides_by_hour(df, key):
    df_hr = df[['hour', key, 'net_checkout']].pivot_table(index=key, columns='hour',
                                                          values='net_checkout',
                                                          aggfunc=np.mean)

    df_hr.columns = [str(int(x)) for x in df_hr.columns.tolist()]
    df_hr.reset_index(inplace=True)
    return df_hr


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    hierarchy.dendrogram(linkage_matrix, **kwargs)


def main(run_eda):
    use_trim = True
    update_data = True

    root_path = os.getcwd()
    cache_path = os.path.join(root_path, r'data/202008-citibike-tripdata-trimmed.pickle')
    cache_path_full = os.path.join(root_path, r'data/202008-citibike-tripdata-full.pickle')

    if (update_data == False) and (use_trim):
        data = pickle.load(open(cache_path, 'rb'))
        print(f'Loaded trimmed data from {cache_path}')
    elif (update_data == False) and (use_trim == False):
        data = pickle.load(open(cache_path_full, 'rb'))
    else:
        data_JC = pd.read_csv(os.path.join(root_path, r'data/JC-202008-citibike-tripdata.csv'))
        data_NY = pd.read_csv(os.path.join(root_path, r'data/202008-citibike-tripdata.csv'))
        print(f'Loaded full data')
        if use_trim:
            data_NY_part = data_NY[::100]
            # data=data_process_data(pd.concat([data_JC, data_NY_part]))
            data = data_process_data(copy.deepcopy(data_NY_part))
            pickle.dump(data, open(cache_path, 'wb'))
            print(f'Use trim data, saved a cache into {cache_path}')
        else:
            data_NY_part = data_NY[::10]
            # data=data_process_data(pd.concat([data_JC, data_NY_part]))
            data = data_process_data(copy.deepcopy(data_NY_part))
            pickle.dump(data, open(cache_path_full, 'wb'))

            print(f'Use full data, saved a cache into {cache_path_full}')

    mask = data[data['start station id'] == data['end station id']].index
    data = data.drop(index=mask)

    map_JC = gpd.read_file(
        os.path.join(root_path, r'Data/jersey-city-neighborhoods/jersey-city-neighborhoods.shp')).to_crs(epsg=4326)
    map_JC = map_JC[['name', 'geometry']]
    map_JC['name'] = map_JC['name'].apply(lambda x: f'JC {x}')
    map_JC['region'] = 'JC'
    map_JC.columns = ['area', 'geometry', 'boro']
    map_NY = gpd.read_file(os.path.join(root_path, r'Data/Neighborhood Tabulation Areas/NY neighborhoods.shp')).to_crs(
        epsg=4326)
    map_NY = map_NY[['ntaname', 'geometry', 'boro_name']]
    map_NY.columns = ['area', 'geometry', 'boro']
    map = pd.concat([map_JC, map_NY], ignore_index=True)
    map['centroid'] = map.geometry.centroid

    # EDA
    run_eda = False
    if run_eda:
        plt.close('all')
        data['start_hr'].value_counts(sort=False).plot(kind='bar')
        data['start_weekday'].value_counts(sort=False).plot(kind='bar')
        data['usertype'].value_counts(sort=False).plot(kind='bar')
        data.groupby('usertype')['start_weekday'].value_counts(sort=False).plot(kind='bar')
        data.groupby('usertype')['start_hr'].value_counts(sort=False).plot(kind='bar')
        ax = data[data['usertype'] == 'Subscriber'].groupby(['start_weekday'])['start_hr'].value_counts(
            sort=False).plot(kind='bar')
        data[data['usertype'] == 'Customer'].groupby(['start_weekday'])['start_hr'].value_counts(sort=False).plot(
            kind='bar', ax=ax, color='red')
        ax.xaxis.set_major_locator(ticker.NullLocator())
        # Outlier on the first two days - need to remove

    # get map and station info with area
    station_profile = summarize_station(data)
    station_profile = gpd.GeoDataFrame(station_profile,
                                       geometry=gpd.points_from_xy(station_profile['station longitude'],
                                                                   station_profile['station latitude']),
                                       crs={'init': 'epsg:4326', 'no_defs': True})

    station_profile_gis = gpd.sjoin(station_profile, map, how='left', op='within')

    # summarize net rides by station by hour
    data = pd.merge(data, area_concat('start', station_profile_gis[['station id', 'area', 'boro']]), how='left',
                    on='start station id')
    data = pd.merge(data, area_concat('end', station_profile_gis[['station id', 'area', 'boro']]), how='left',
                    on='end station id')

    # group by station
    rides_byStation = summary_ride(data, 'station id')
    rides_byStation_byHour = summarize_rides_by_hour(rides_byStation, 'station id')

    # group by area
    rides_byArea = summary_ride(data, 'area')
    # len(rides_byArea[rides_byArea['net_checkout'].apply(lambda x: isinstance(x, float)==False)])
    rides_byArea_byHour = summarize_rides_by_hour(rides_byArea, 'area')
    # rides_byArea_byHour_gis=gpd.GeoDataFrame(rides_byArea_byHour.merge(map[['boro','area','centroid']], on='area'), geometry='centroid')
    rides_byArea_byHour_gis = gpd.GeoDataFrame(rides_byArea_byHour.merge(map[['boro', 'area', 'geometry']], on='area'),
                                               geometry='geometry')

    plot_rides_on_map = False
    if plot_rides_on_map:
        rides_byStation_byHour_gis = pd.merge(rides_byStation_byHour, station_profile_gis, on='station id')
        for i in range(0, 24):
            ax = map.plot(figsize=(8, 8), alpha=0.5, edgecolor='k')
            # rides_byStation_byHour_gis.plot(ax=ax, color='red', markersize=rides_byStation_byHour_gis[0])
            select_hr = str(i)
            gplt.pointplot(rides_byStation_byHour_gis[[select_hr, 'geometry']], hue=select_hr, scale=select_hr, ax=ax,
                           legend=True, legend_var='hue')
            plt.savefig(os.path.join(root_path, r'plots/202008_station_' + select_hr + '.png'))
            # lda/pca to reduce features
            plt.close('all')

        for i in range(0, 24):
            ax = map.plot(figsize=(8, 8), alpha=0.5, edgecolor='k')

            # rides_byStation_byHour_gis.plot(ax=ax, color='red', markersize=rides_byStation_byHour_gis[0])
            select_hr = str(i)
            # gplt.pointplot(rides_byArea_byHour_gis[[select_hr, 'centroid']], hue=select_hr, scale=select_hr, ax=ax, legend=True,
            #                legend_var='hue')
            rides_byArea_byHour_gis.plot(column=select_hr, ax=ax, legend=True)

            plt.savefig(os.path.join(root_path, r'plots/202008_area_choropleth_' + select_hr + '.png'))
            plt.close('all')

    data['distance'] = abs(data['end station longitude'] - data['start station longitude']) + abs(
        data['end station latitude'] - data['start station latitude'])
    data.drop(index=data[data['distance'] == 0].index, inplace=True)
    data['speed'] = data['distance'] / data['tripduration']
    # data['start_area_net_checkout'] = data[['start area','start_date_hr']].apply(
    #     lambda x: rides_byArea[((rides_byArea['area']==x.iloc[0]) & (rides_byArea['date_hour'] == x.iloc[1]))]['net_checkout'])
    start_area_checkout = rides_byArea[['area', 'date_hour', 'net_checkout']]
    start_area_checkout.columns = ['start area', 'start_date_hr', 'start_area_net_checkout']
    data = pd.merge(data, start_area_checkout, on=['start area', 'start_date_hr'], how='left')
    end_area_checkout = rides_byArea[['area', 'date_hour', 'net_checkout']]
    end_area_checkout.columns = ['end area', 'stop_date_hr', 'end_area_net_checkout']
    data = pd.merge(data, end_area_checkout, on=['end area', 'stop_date_hr'], how='left')

    start_station_checkout = rides_byStation[['station id', 'date_hour', 'net_checkout']]
    start_station_checkout.columns = ['start station id', 'start_date_hr', 'start_station_net_checkout']
    data = pd.merge(data, start_station_checkout, on=['start station id', 'start_date_hr'], how='left')

    end_station_checkout = rides_byStation[['station id', 'date_hour', 'net_checkout']]
    end_station_checkout.columns = ['end station id', 'stop_date_hr', 'end_station_net_checkout']
    data = pd.merge(data, end_station_checkout, on=['end station id', 'stop_date_hr'], how='left')

    feature_visualization = False
    plt.close('all')
    if feature_visualization:
        sns.distplot(data['start station latitude'])
        sns.distplot(data['start station longitude'])
        sns.distplot(data.start_area_net_checkout)
        sns.distplot(data.end_area_net_checkout)
        sns.distplot(data.start_station_net_checkout)
        sns.distplot(data.end_station_net_checkout)
        sns.distplot(data.distance)
        sns.distplot(data['distance'].apply(lambda x: math.log(x * 100)))
        sns.distplot(data.speed)

    # customer feature normalization
    data_customer_std = pd.DataFrame()
    data_customer_std['hr_x'] = data['start_hr'].apply(lambda hour: math.sin(2 * math.pi * hour / 24))
    data_customer_std['hr_y'] = data['start_hr'].apply(lambda hour: math.cos(2 * math.pi * hour / 24))
    col = 'distance'
    data_customer_std[col] = data[col].apply(lambda x: math.log(x * 100))
    # col='start_weekday'
    # data_customer_std[col]= data[col].apply(lambda x: 1 if x>=6 else 0)
    data_customer_std['weekday_x'] = data['start_weekday'].apply(lambda day: math.sin(2 * math.pi * day / 7))
    data_customer_std['weekday_y'] = data['start_weekday'].apply(lambda day: math.cos(2 * math.pi * day / 7))

    col_list = ['distance',
                'start station latitude', 'start station longitude',
                'end station latitude', 'end station longitude',
                'start_area_net_checkout',
                'end_area_net_checkout',
                'start_station_net_checkout',
                'end_station_net_checkout']
    data_customer_std.loc[:, col_list] = data[col_list]
    data_customer_std.fillna(0, inplace=True)
    for col in data_customer_std.columns:
        data_customer_std[col] = data_customer_std[col] / np.std(data_customer_std[col])
    # sns.violinplot(data=data_customer_std,orient='h')

    dimension_reduction = False
    ## dimension reduction for visualization
    if dimension_reduction:
        # pca
        pca_plot = True
        if pca_plot:
            pca = PCA()
            data_customer_pca = pca.fit_transform(data_customer_std)
            fig = plt.figure(figsize=(12, 8))
            plt.scatter(data_customer_pca[:, 0], data_customer_pca[:, 1], s=1)
            plt.xlabel('pca feature 1')
            plt.ylabel('pca feature 2')
            plt.title('pca dimension reduction 2D')

            # pca.explained_variance_
            # pca_components=pd.DataFrame(pca.components_)
            # pca_components.columns=data_customer.columns
            # ax = pca_components.plot(kind='bar',stacked=True)
            # ax.legend(loc=1,fontsize=8)
            plt.savefig(os.path.join(root_path, r'plots/202008_pca_2D.png'))

        tsne_plot = False
        if tsne_plot:
            # t-SNE
            tsne = TSNE(random_state=42, n_components=3, verbose=0, perplexity=40, n_iter=400).fit_transform(
                data_customer_std)
            # 2D
            fig = plt.figure(figsize=(12, 8))
            plt.scatter(tsne[:, 0], tsne[:, 1], s=1)
            plt.xlabel('tsne feature 1')
            plt.ylabel('tsne feature 2')
            plt.title('tSNE dimension reduction 2D')
            plt.savefig(os.path.join(root_path, r'plots/202008_tsne_2D.png'))
            # 3D
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(tsne[:, 0], tsne[:, 1], tsne[:, 2], s=1)
            ax.set_xlabel('tsne feature 1')
            ax.set_ylabel('tsne feature 2')
            ax.set_zlabel('tsne feature 3')
            plt.title('tSNE dimension reduction 3D')
            plt.savefig(os.path.join(root_path, r'plots/202008_tsne_3D.png'))
            plt.close('all')

        # umap
        umap_plot = False
        if umap_plot:
            fig, ax = plt.subplots(3, 2)
            fig.set_size_inches(10, 20)
            for i, n in enumerate([10, 50, 100]):
                embedding_corr = umap.UMAP(n_neighbors=n,
                                           min_dist=0.3,
                                           metric='correlation').fit_transform(data_customer_std)

                ax[i, 0].scatter(embedding_corr[:, 0], embedding_corr[:, 1],
                                 edgecolor='none',
                                 alpha=0.80,
                                 s=10)
                ax[i, 0].set_xlabel('umap feature 1')
                ax[i, 0].set_ylabel('umap feature 2')
                ax[i, 0].set_title(f'umap dimension reduction_corr metrics_{n}_neighbors')

                embedding_dist = umap.UMAP(n_neighbors=n,
                                           min_dist=0.3,
                                           metric='euclidean').fit_transform(data_customer_std)

                ax[i, 1].scatter(embedding_dist[:, 0], embedding_dist[:, 1],
                                 edgecolor='none',
                                 alpha=0.80,
                                 s=10)
                ax[i, 1].set_xlabel('umap feature 1')
                ax[i, 1].set_ylabel('umap feature 2')
                ax[i, 1].set_title(f'umap dimension reduction_euclidean metrics_{n}_neighbors')

            plt.suptitle('umap visualization')
            plt.savefig(os.path.join(root_path, r'plots/202008_umap_visualization.png'))
            plt.close('all')

    clustering = True
    if clustering:
        ## clustering
        # k-means
        data_customer_std_sample = copy.deepcopy(data_customer_std.loc[::1, :])
        num_max = 4
        clustering_kmeans = True
        if clustering_kmeans:
            start_time = time.process_time()
            kmeans_labels_agg = {}
            sil_scores_kmeans_agg = {}
            ch_scores_kmeans_agg = {}
            for num in range(2, num_max + 1):
                kmeans = KMeans(n_clusters=num, random_state=0)
                kmeans_labels_agg[num] = kmeans.fit_predict(data_customer_std_sample)
                sil_scores_kmeans_agg[num] = metrics.silhouette_score(data_customer_std_sample, kmeans_labels_agg[num])
                ch_scores_kmeans_agg[num] = metrics.calinski_harabasz_score(data_customer_std_sample,
                                                                            kmeans_labels_agg[num])
            # pd.DataFrame.from_dict(sil_scores_kmeans_agg.values()).plot()

        clustering_hierachy = True
        if clustering_hierachy:
            start_time = time.process_time()
            ward_labels_agg = {}
            sil_scores_ward_agg = {}
            ch_scores_ward_agg = {}
            for num in range(2, num_max + 1):
                ward_clustering = AgglomerativeClustering(n_clusters=num, linkage='ward').fit(data_customer_std_sample)
                ward_labels_agg[num] = ward_clustering.labels_

                sil_scores_ward_agg[num] = metrics.silhouette_score(data_customer_std_sample, ward_labels_agg[num])
                ch_scores_ward_agg[num] = metrics.calinski_harabasz_score(data_customer_std_sample,
                                                                          ward_labels_agg[num])
                print(f'ward clustering takes time {time.process_time() - start_time}')
            # pd.DataFrame.from_dict(sil_scores_ward_agg.values()).plot()

        clustering_gmm = True
        if clustering_gmm:
            start_time = time.process_time()
            gmm_labels_agg = {}
            sil_scores_gmm_agg = {}
            ch_scores_gmm_agg = {}
            for num in range(2, num_max + 1):
                gmm_clustering = GaussianMixture(n_components=num).fit(data_customer_std_sample)
                gmm_labels_agg[num] = gmm_clustering.predict(data_customer_std_sample)
                sil_scores_gmm_agg[num] = metrics.silhouette_score(data_customer_std_sample, gmm_labels_agg[num])
                ch_scores_gmm_agg[num] = metrics.calinski_harabasz_score(data_customer_std_sample, gmm_labels_agg[num])
                print(f'gmm clustering takes time {time.process_time() - start_time}')

        umap_clustering = True
        if umap_clustering:
            embedding_corr = umap.UMAP(n_neighbors=10,
                                       min_dist=0.3,
                                       metric='correlation').fit_transform(data_customer_std_sample)

            start_time = time.process_time()
            kmeans_labels_umap = {}
            sil_scores_kmeans_umap = {}
            ch_scores_kmeans_umap = {}
            for num in range(2, num_max + 1):
                kmeans = KMeans(n_clusters=num, random_state=0)
                kmeans_labels_umap[num] = kmeans.fit_predict(embedding_corr)
                sil_scores_kmeans_umap[num] = metrics.silhouette_score(data_customer_std_sample,
                                                                       kmeans_labels_umap[num])
                ch_scores_kmeans_umap[num] = metrics.calinski_harabasz_score(data_customer_std_sample,
                                                                             kmeans_labels_umap[num])

            start_time = time.process_time()
            ward_labels_umap = {}
            sil_scores_ward_umap = {}
            ch_scores_ward_umap = {}
            for num in range(2, num_max + 1):
                ward_clustering = AgglomerativeClustering(n_clusters=num, linkage='ward').fit(embedding_corr)
                ward_labels_umap[num] = ward_clustering.labels_
                sil_scores_ward_umap[num] = metrics.silhouette_score(data_customer_std_sample, ward_labels_umap[num])
                ch_scores_ward_umap[num] = metrics.calinski_harabasz_score(data_customer_std_sample,
                                                                           ward_labels_umap[num])
                print(f'ward clustering takes time {time.process_time() - start_time}')

            start_time = time.process_time()
            gmm_labels_umap = {}
            sil_scores_gmm_umap = {}
            ch_scores_gmm_umap = {}
            for num in range(2, num_max + 1):
                gmm_clustering = GaussianMixture(n_components=3).fit(embedding_corr)
                gmm_labels_umap[num] = gmm_clustering.predict(embedding_corr)
                sil_scores_gmm_umap[num] = metrics.silhouette_score(data_customer_std_sample, gmm_labels_umap[num])
                ch_scores_gmm_umap[num] = metrics.calinski_harabasz_score(data_customer_std_sample,
                                                                          gmm_labels_umap[num])
                print(f'gmm clustering takes time {time.process_time() - start_time}')

        plot_hierachy_linkage = False
        if plot_hierachy_linkage:
            ward_clustering_full = AgglomerativeClustering(distance_threshold=0, n_clusters=None).fit(
                data_customer_std_sample)
            linkage = hierarchy.linkage(ward_clustering_full.children_, 'ward')
            plt.figure(figsize=(10, 7))
            dn = hierarchy.dendrogram(linkage)
            # plot_dendrogram(ward_clustering_full,truncate_mode='level', p=3)

        plot_clustering_2D = False
        if plot_clustering_2D:
            embedding_corr = umap.UMAP(n_neighbors=10,
                                       min_dist=0.3,
                                       metric='correlation').fit_transform(data_customer_std_sample)

            # labels=ward_labels_agg[4]
            labels = ward_labels_umap[2]
            # visualize clustering
            fig = plt.figure(figsize=(12, 8))
            plt.scatter(embedding_corr[:, 0], embedding_corr[:, 1],
                        edgecolor='none',
                        alpha=0.80,
                        s=10,
                        c=labels)
            plt.xlabel('umap feature 1')
            plt.ylabel('umap feature 2')
            # plt.title(f'umap visualization with kmeans clustering labelling')
            # plt.savefig(os.path.join(root_path,r'plots/202008_umap_visualization_kmeans_clustering.png'))

            plt.title(f'umap visualization with ward hierachy clustering labelling')
            plt.savefig(os.path.join(root_path, r'plots/202008_umap_visualization_ward_clustering.png'))

        plot_clustering_feature_detail = True
        if plot_clustering_feature_detail:
            # analyze feature importance
            labels_dict = {0: kmeans_labels_agg,
                           1: ward_labels_agg,
                           2: gmm_labels_agg,
                           3: kmeans_labels_umap,
                           4: ward_labels_umap,
                           5: gmm_labels_umap}
            labels_str_dict = {0: 'kmeans',
                               1: 'ward',
                               2: 'gmm',
                               3: 'kmeans_umap',
                               4: 'ward_umap',
                               5: 'gmm_umap'}

            for type in range(0, 3):
                for cluster_num in range(2, num_max + 1):
                    # cluster_num=4
                    col_select = ['start station longitude', 'start station latitude',
                                  'end station latitude', 'end station longitude',
                                  'start_hr', 'start_weekday',
                                  'start_area_net_checkout', 'end_area_net_checkout',
                                  'start_station_net_checkout', 'end_station_net_checkout'
                                  ]
                    # fig, ax = plt.subplots(len(col_select), cluster_num)
                    # fig.set_size_inches(5 * cluster_num, 20)
                    # plt.suptitle('clustering feature analysis')
                    # plt.tight_layout()
                    #
                    # labels = labels_dict[type][cluster_num]
                    # df_customer_cluster = {}
                    #
                    #
                    # for cluster_i in range(0, cluster_num):
                    #     print(f'analyze cluster {cluster_i}')
                    #     mask_i = np.argwhere(labels == cluster_i).ravel()
                    #     mask_i_original = data_customer_std_sample.iloc[mask_i].index
                    #     df_customer_cluster[cluster_i] = data.loc[mask_i_original].copy()
                    #     for i, col in enumerate(col_select):
                    #         ax[i, cluster_i] = sns.histplot(ax=ax[i, cluster_i],
                    #                                         data=df_customer_cluster[cluster_i][col], kde=True)
                    #
                    # plt.savefig(os.path.join(root_path, r'plots',
                    #                          f'202008_clustering feature analysis_{labels_str_dict[type]}_{cluster_num}.png'))

                    labels = labels_dict[type][cluster_num]
                    df_customer_cluster = {}

                    fig, ax = plt.subplots(4, cluster_num)
                    fig.set_size_inches(8*cluster_num, 15)
                    plt.tight_layout(pad=8)
                    for cluster_i in range(0, cluster_num):
                        print(f'analyze cluster {cluster_i} of {labels_str_dict[type]}')
                        mask_i = np.argwhere(labels == cluster_i).ravel()
                        mask_i_original = data_customer_std_sample.iloc[mask_i].index
                        df_customer_cluster[cluster_i] = copy.deepcopy(data.loc[mask_i_original])
                        df_customer_cluster[cluster_i]['start area']=df_customer_cluster[cluster_i]['start area'].apply(lambda x:x.split('-')[0])
                        df_customer_cluster[cluster_i]['end area']=df_customer_cluster[cluster_i]['end area'].apply(lambda x:x.split('-')[0])

                        #bar plot starting area
                        pd.DataFrame((df_customer_cluster[cluster_i]['start area'].value_counts())).sort_values(
                            by='start area', ascending=False).head(20).plot(kind='barh', ax=ax[0, cluster_i])
                        ax[0, cluster_i].title.set_text(f'Top 20 Start Area in cluster {cluster_i+1}')
                        #bar plot end area
                        pd.DataFrame((df_customer_cluster[cluster_i]['end area'].value_counts())).sort_values(
                            by='end area', ascending=False).head(20).plot(kind='barh', ax=ax[1, cluster_i])
                        ax[1, cluster_i].title.set_text(f'Top 20 End Area in cluster {cluster_i+1}')
                        #bar plot riding time
                        df_start_time_raw=df_customer_cluster[cluster_i][['start_weekday', 'start_hr']].groupby(
                            ['start_weekday', 'start_hr']).size()
                        df_start_time=df_start_time_raw.reset_index()
                        df_start_time.columns=['ride_day','ride_hr','count']
                        sns.histplot(df_start_time, x="ride_hr",binwidth=1, y='count',hue='ride_day',ax=ax[2, cluster_i])
                        ax[2, cluster_i].title.set_text(f'Day/Time of the rides in cluster {cluster_i+1}')

                        #bar plot start and end area demand comparison
                        df_val = df_customer_cluster[cluster_i][
                            ['start_area_net_checkout', 'end_area_net_checkout']].groupby(
                            ['start_area_net_checkout', 'end_area_net_checkout']).size()
                        df_checkout = df_customer_cluster[cluster_i][
                            ['start_area_net_checkout', 'end_area_net_checkout']].copy()
                        df_checkout.dropna(inplace=True)
                        df_checkout['val'] = df_checkout.apply(lambda x: df_val[x.iloc[0]][x.iloc[1]], axis=1)
                        df_checkout.plot.scatter(
                            x='start_area_net_checkout', y='end_area_net_checkout', s='val', ax=ax[3, cluster_i])
                        ax[3, cluster_i].title.set_text(f'Net Checkouts Comparison of Start & End Area in {cluster_i+1}')

                        plt.setp(ax[0, cluster_i].yaxis.get_majorticklabels(), fontsize=8)
                        plt.setp(ax[1, cluster_i].yaxis.get_majorticklabels(), fontsize=8)
                        plt.setp(ax[2, cluster_i].yaxis.get_majorticklabels(), fontsize=8)
                    plt.savefig(
                        os.path.join(root_path, r'plots',
                                     f'202008_{labels_str_dict[type]}_{cluster_num}_cluster_feature_detail.png'))

                    plt.close('all')

            for type in range(0, 3):
                for cluster_num in range(2, num_max + 1):
                    # geoplot
                    # type = 2
                    # cluster_num = 3
                    labels = labels_dict[type][cluster_num]
                    df_customer_cluster = {}
                    fig, ax = plt.subplots(cluster_num, 2)
                    fig.set_size_inches(15, 7*cluster_num)
                    plt.tight_layout(pad=5)

                    for cluster_i in range(0, cluster_num):
                        print(f'analyze cluster {cluster_i}')
                        mask_i = np.argwhere(labels == cluster_i).ravel()
                        mask_i_original = data_customer_std_sample.iloc[mask_i].index
                        df_customer_cluster[cluster_i] = data.loc[mask_i_original].copy()
                        # df_customer_cluster[cluster_i] = pd.merge(df_customer_cluster[cluster_i],
                        #                                           area_concat('start', station_profile_gis[
                        #                                               ['station id', 'geometry']]),
                        #                                           how='left', on='start station id')
                        # df_customer_cluster[cluster_i] = pd.merge(df_customer_cluster[cluster_i],
                        #                                           area_concat('end', station_profile_gis[
                        #                                               ['station id', 'geometry']]),
                        #                                           how='left', on='end station id')

                        counter_start = df_customer_cluster[cluster_i]['start area'].value_counts()
                        counter_end = df_customer_cluster[cluster_i]['end area'].value_counts()
                        df_customer_cluster[cluster_i]['start_area_count'] = df_customer_cluster[
                            cluster_i]['start area'].apply(lambda x: counter_start[x])
                        df_customer_cluster[cluster_i]['end_area_count'] = df_customer_cluster[
                            cluster_i]['end area'].apply(lambda x: counter_end[x])


                        df_start = gpd.GeoDataFrame(df_customer_cluster[cluster_i],
                                                           geometry=gpd.points_from_xy(df_customer_cluster[cluster_i]['start station longitude'],
                                                                                       df_customer_cluster[cluster_i]['start station latitude']),
                                                           crs={'init': 'epsg:4326', 'no_defs': True})

                        map.plot(ax=ax[cluster_i,0],figsize=(8, 8), alpha=0.5, edgecolor='k')
                        gplt.pointplot(df_start, hue='start_hr', scale='start_area_count',
                                       legend=True, legend_var='hue',ax=ax[cluster_i,0])

                        df_end = gpd.GeoDataFrame(df_customer_cluster[cluster_i],
                                                    geometry=gpd.points_from_xy(
                                                        df_customer_cluster[cluster_i]['end station longitude'],
                                                        df_customer_cluster[cluster_i]['end station latitude']),
                                                    crs={'init': 'epsg:4326', 'no_defs': True})

                        map.plot(ax=ax[cluster_i,1],figsize=(8, 8), alpha=0.5, edgecolor='k')
                        gplt.pointplot(df_end, hue='stop_hr', scale='end_area_count',
                                       legend=True, legend_var='hue',ax=ax[cluster_i,1])
                        ax[cluster_i,0].title.set_text(f'Start Station Net Checkouts in cluster {cluster_i + 1}')
                        ax[cluster_i,1].title.set_text(f'End Station Net Checkouts in cluster {cluster_i + 1}')
                    plt.savefig(os.path.join(root_path, r'plots',f'202008_{labels_str_dict[type]}_{cluster_num}_station_detail.png'))

                    plt.close('all')



    run_classification = True
    if run_classification:
        y = gmm_labels_agg[3]

        from sklearn.model_selection import train_test_split
        x_train, x_test, y_train, y_test = train_test_split(data_customer_std_sample, y, test_size=0.2)

        from sklearn.naive_bayes import GaussianNB
        from sklearn.metrics import accuracy_score
        gnb = GaussianNB()
        gnb.fit(x_train, y_train)
        y_predict = gnb.predict(x_test)
        y_combo = list(zip(y_test, y_predict))
        accuracy_score(y_test, y_predict)


if __name__ == '__main__':
    run_eda = True
    main(run_eda)
