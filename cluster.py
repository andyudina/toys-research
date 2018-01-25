# -*- coding: utf-8 -*-
"""
Find personas from filtered sextoys
"""
import math

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

from kmodes.kprototypes import KPrototypes
from kmodes.kmodes import KModes

from utils import flatten_multiple_choices, is_nan

df = pd.read_csv('sextoys_cleaned.csv')

simple_categorical_features = [
   'Gender',
   'Sexuality',
   'Relationship status',
   'How often do you buy sex toys for yourself?']

multiple_categorical_features = [
   'sources_of_discomfort',
   'not_buy_reasons']

numeric_features = [
    'I use sex toys regularly in my sex live',
    'I use sex toys with my partners',
    'I’m adventurous when it comes to trying new things',
    'I feel discomfort when buying sex toys'
]

def preprocess_columns_for_feature_binarisation(
        df, 
        simple_categorical_features, 
        multiple_categorical_features,
        numeric_features):
    """
    Clean column names.
    Flatten feature with multiple choices
    """
    final_columns = multiple_categorical_features[:]
    for feature in numeric_features + simple_categorical_features:
        new_column = feature.strip().lower().replace(' ', '_')
        df[new_column] = df[feature]
        final_columns.append(new_column)
    for feature in multiple_categorical_features:
        df = flatten_multiple_choices(df, feature)
    categorical_columns = [
        x.lower().replace(' ', '_') 
        for x in simple_categorical_features] + \
        multiple_categorical_features
    return df, final_columns, categorical_columns

df, final_columns, categorical_columns = preprocess_columns_for_future_binarisation(
        df, 
        simple_categorical_features, 
        multiple_categorical_features,
        numeric_features)

def get_silhouette_score(
        df, labels, categorical_columns, numeric_columns):
    """
    Binarise categorical_columns and calculate silhoutte score
    """
    result_columns = []
    for column in categorical_columns:
        for val in df[column].unique():
            new_column = '--'.join([column, str(val)])
            df[new_column] = df.apply(
                lambda x: x[column] == val,
                axis=1)
            result_columns.append(new_column)
    X = df[result_columns + numeric_columns].as_matrix()
    return silhouette_score(
        X, labels, metric='euclidean')#  Better -Gower's distance

def try_clusterise_to_number_of_clusters(
        df, n_clusters, columns,
        categorical_columns):
    """
    Use Kmodes (or Kprototypes, if there are numerical clusters) to find
    n_clusters in given dataset
    """
    df_ = df[columns].dropna(axis=1, how='all')
    df_.fillna(0, inplace=True)
    X = df_.as_matrix()
    categorical_columns_numbers = []
    for index, column in enumerate(df_.columns.values):
        if column in categorical_columns:
            categorical_columns_numbers.append(index)
    if len(df_.columns.values) == len(categorical_columns_numbers):
        class_ = KModes
    else:
        class_ = KPrototypes
    k = class_(n_clusters=n_clusters)
    clusters = k.fit(X, categorical=categorical_columns_numbers)
    labels = clusters.labels_
    score = get_silhouette_score(
        df_, labels, 
        list(set(categorical_columns) & set(df_.columns.values)),
        list(set(df_.columns.values) - set(categorical_columns)))
    return {
        'cost': k.cost_,
        'centr': k.cluster_centroids_,
        'score': score,
        'labels': labels,
        'n_clusters': n_clusters
    }

# split dataset to two different datasets: people who use toys
# and people who do not use toys. Each of group has different quiestions
# in questionare and thus different set of categorical features
df_not_use = df[df.not_buy_reasons.notnull()]
df_use = df[df.sources_of_discomfort.notnull()]

def try_find_clusters(
        df, max_number_of_clusters, 
        columns, categorical_columns):
    """
    Try find clusters wich will maximise silhuette score on given dataset
    Print out results to stdout
    """
    results = []
    for n_clusters in xrange (2, max_number_of_clusters):
        res = try_clusterise_to_number_of_clusters(
            df, n_clusters, columns, categorical_columns)
        results.append(res)
        #print(res)

    for r in sorted(
            results, key=lambda x: x['score'], reverse=True)[:3]:
        print (r['score'], r['n_clusters'], r['centr'])        
        print
    print

try_find_clusters(df_not_use, 13, final_columns, categorical_columns)

try_find_clusters(df_use, 20, final_columns, categorical_columns)
