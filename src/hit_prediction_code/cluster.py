"""Module containing functions to select features based on a clustering."""
from logzero import logger
import pandas as pd
from sklearn import cluster


def get_cluster_song_count(data,
                           cluster_id='cluster_id',
                           song_id='song_id',
                           feature_id='feature_id'):
    """Count how many song samples are in which cluster.

    Args:
        data: a pandas dataframe containing cluster_id, song_id and feature_id.
        cluster_id: the name of the cluster_id column.
        song_id: the name of the song_id column.
        feature_id: the name of the feature_id column.

    Returns a dataframe containing the counts.
    """
    grouping_cols = [cluster_id, song_id]
    relevant_cols = grouping_cols + [feature_id]

    count = data[relevant_cols].groupby(by=grouping_cols).count().reset_index()
    count.rename(columns={feature_id: 'count'}, inplace=True)

    return count


def select_cluster_song_features(data,
                                 certainty=2,
                                 cluster_id='cluster_id',
                                 song_id='song_id',
                                 feature_id='feature_id'):
    """Removes features that are in a different cluster than the majority.

    Args:
        data: a pandas dataframe containing cluster_id, song_id and feature_id.
        certainty: used to define the threshold for dropping features. If the
            song count in a cluster is smaller than the maximum song count in a
            cluster divided by this certainty than a sample is dropped.
        cluster_id: the name of the cluster_id column.
        song_id: the name of the song_id column.
        feature_id: the name of the feature_id column.

    Returns the dataframe with the outlier removed.
    """
    counts = get_cluster_song_count(
        data=data,
        cluster_id=cluster_id,
        song_id=song_id,
        feature_id=feature_id,
    )

    join_cols = [cluster_id, song_id]

    keep = pd.DataFrame()
    for _, group in counts.groupby(song_id):
        keep_group = group[group['count'] >= group['count'].max() / certainty]
        keep = keep.append(keep_group[join_cols])

    return data.merge(keep, on=join_cols)


def generate_song_id(data, song_id_cols):
    """Generates ids for unique songs identified by the given columns.

    Args:
        data: dataframe containing the song.
        song_id_cols: columns identifying a unique song.

    Returns a dataframe containing a mapping between the id and the identifying
    columns.
    """
    songs = data[song_id_cols].drop_duplicates().reset_index()
    del songs['index']
    songs['song_id'] = songs.index

    return songs


def select_features_with_clustering(data,
                                    song_id_cols,
                                    feature_cols,
                                    max_iterations=10,
                                    certainty=2,
                                    feature_id='feature_id'):
    """Selects features based on feature clusters.

    Args:
        data: dataframe containing the dataset.
        song_id_cols: list of collumns identifying a song.
        feature_cols: columns used to build the clusters.
        max_iterations: the maximum number of iterations used to drop features.

    Returns a dataframe without the selected samples.
    """
    songs = generate_song_id(data=data, song_id_cols=song_id_cols)
    data = data.merge(songs, on=song_id_cols)

    for i in range(1, max_iterations + 1):
        data_size = len(data)
        logger.debug('Iteration %d processing %d features.' % (i, data_size))
        clustering = cluster.AgglomerativeClustering(
            n_clusters=len(songs)).fit(data[feature_cols])
        data['cluster_id'] = clustering.labels_

        data = select_cluster_song_features(
            data,
            certainty=certainty,
            feature_id=feature_id,
        )

        if data_size == len(data):
            logger.debug('Fix point reached after %d iterations.' % i)
            break

    return data
