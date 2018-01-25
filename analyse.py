# -*- coding: utf-8 -*-
"""
Visualise answers distribution
Find correlations btw features
Find stats for different demographic groups
"""
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from utils import flatten_multiple_choices, is_nan


matplotlib.style.use('ggplot')
plt.figure()

df = pd.read_csv('sextoys_cleaned.csv')

simple_categorical_features = [
   'Country',
   'Gender',
   'Sexuality',
   'Relationship status',
   'Education',
   'How often do you buy sex toys for yourself?']

multiple_categorical_features = [
   'sources_of_discomfort',
   'not_buy_reasons']

numeric_features = [
    'I use sex toys regularly in my sex live',
    'I use sex toys with my partners',
    'I\'m adventurous when it comes to trying new things',
    'I feel discomfort when buying sex toys'
]


def visualize_distribution(
       metric, df, rotation=45):
    """
    Plot distribution of metric
    """
    sequence = df[metric]
    sequence.value_counts(normalize=True).sort_index().plot(
        title=metric,
        kind='bar')
    plt.xticks(rotation=rotation)
    plt.show()

# Flatten multiple choices fields to new dataframe
df_flattened = df.copy()
for feature in multiple_categorical_features:
    df_flattened = flatten_multiple_choices(
        df_flattened, feature)


def clean_gender(gender):
    """
    Clean gender into 3 categoris
    """
    if gender in ['female', 'male']:
        return gender
    return 'other'


def clean_sex(row):
    """
    Clean sexuality into 3 categories: hetero, homo, others
    """
    if row.Sexuality in ['heterosexual', 'homosexual']:
        return row.Sexuality
    if row.Sexuality == 'bisexual':
        if row.Gender == 'male':
            return 'homosexual'
        elif row.Gender == 'female':
            return 'heterosexual'
        else: return 'other'
    return 'other'

def clean_partner(partner):
    """
    Clean relationship status into 3 categories: with, without, others
    """
    if partner in [
        'with partner', 'married', 
        'with partners / poly relationship / open relationship',
        'married with open relationship']:
        return 'with_partner'
    elif partner in [
        'single - dating',
        'single - not dating'
        'recently divorced']:
        return 'without_partner'
    else:
        return 'other'

def clean_significant_demographics(df):
    """
    Clean significant (with not uniformed distribution) demographic features
    """
    df['gender_cleaned'] = df.apply(
        lambda x: clean_gender(x['Gender']),
        axis=1)

    df['sexuality_cleaned'] = df.apply(
        lambda x: clean_sex(x),
        axis=1)

    df['partner_cleaned'] = df.apply(
        lambda x: clean_partner(x['Relationship status']),
        axis=1)

clean_significant_demographics(df)
clean_significant_demographics(df_flattened)

def generate_all_possible_combinations(l_):
    import itertools
    return list(itertools.product(*l_))

def find_most_common_personas():
    """
    Find most common combinations of significant demographic features in the dataset
    """
    possible_personas = generate_all_possible_combinations(
        [
            df['sexuality_cleaned'].unique(),
            df['gender_cleaned'].unique(),
            df['partner_cleaned'].unique()
        ])

    personas_stats = []
    for persona in possible_personas:
        df_persona = df[
            (df.sexuality_cleaned == persona[0]) & \
            (df.gender_cleaned == persona[1]) & \
            (df.partner_cleaned == persona[2])
        ]
        personas_stats.append({
            'description': persona,
            'number': len(df_persona),
            'rate': float(len(df_persona)) / len(df)
        })


    for persona in sorted(
            personas_stats,
            key=lambda x: x['rate'],
            reverse=True
        )[:10]:
        print persona
        print



def print_personas_stats(df, df_flatten):
    """
    Print usage stat for personas
    """
    print(
        df['How often do you buy sex toys for yourself?'].value_counts(normalize=True))
    print
    print(
        df_flatten['sources_of_discomfort'].value_counts(normalize=True))
    print
    print(
        df_flatten['not_buy_reasons'].value_counts(normalize=True))
    print

personas = (
    # females
    (
        'heterosexual, female',
        df[(df.sexuality_cleaned == 'heterosexual') & (df.gender_cleaned == 'female')],
        df_flattened[
            (df_flattened.sexuality_cleaned == 'heterosexual') & \
            (df_flattened.gender_cleaned == 'female')],
    ),
    (
        'heterosexual, female, with_partner',
        df[
            (df.sexuality_cleaned == 'heterosexual') & \
            (df.gender_cleaned == 'female') & \
            (df.partner_cleaned == 'with_partner')],
        df_flattened[
            (df_flattened.sexuality_cleaned == 'heterosexual') & \
            (df_flattened.gender_cleaned == 'female') &\
            (df_flattened.partner_cleaned == 'with_partner')],
    ),
    (
        'heterosexual, female, without_partner',
        df[
            (df.sexuality_cleaned == 'heterosexual') & \
            (df.gender_cleaned == 'female') & \
            (df.partner_cleaned == 'without_partner')],
        df_flattened[
            (df_flattened.sexuality_cleaned == 'heterosexual') & \
            (df_flattened.gender_cleaned == 'female') &\
            (df_flattened.partner_cleaned == 'without_partner')],
    ),

    # males
    (
        'heterosexual, male',
        df[(df.sexuality_cleaned == 'heterosexual') & (df.gender_cleaned == 'male')],
        df_flattened[
            (df_flattened.sexuality_cleaned == 'heterosexual') & \
            (df_flattened.gender_cleaned == 'male')],
    ),
    (
        'homosexual, male',
        df[(df.sexuality_cleaned == 'homosexual') & (df.gender_cleaned == 'male')],
        df_flattened[
            (df_flattened.sexuality_cleaned == 'homosexual') & \
            (df_flattened.gender_cleaned == 'male')],
    ),
    (
        'homosexual, male, with_partner',
        df[
            (df.sexuality_cleaned == 'homosexual') & \
            (df.gender_cleaned == 'male') & \
            (df.partner_cleaned == 'with_partner')],
        df_flattened[
            (df_flattened.sexuality_cleaned == 'homosexual') & \
            (df_flattened.gender_cleaned == 'male') & \
            (df_flattened.partner_cleaned == 'with_partner')],
    ),
    (
        'heterosecual, male, without_partner',
        df[
            (df.sexuality_cleaned == 'heterosexual') & \
            (df.gender_cleaned == 'male') & \
            (df.partner_cleaned == 'without_partner')],
        df_flattened[
            (df_flattened.sexuality_cleaned == 'heterosexual') & \
            (df_flattened.gender_cleaned == 'male') & \
            (df_flattened.partner_cleaned == 'without_partner')],
    ),
)



df_no_need_to_buy_regular = df_flattened[df_flattened['not_buy_reasons'] == 'no_need_to_buy_regular']
def print_stats_for_non_regular(df_no_need_to_buy_regular):
    """
    Print demographic and usage distributions for users, who don't buy regular
    and don't consider themselfs as not regular users
    """
    print(
        'Size',
        len(df_no_need_to_buy_regular),
        float(len(df_no_need_to_buy_regular[
                df_no_need_to_buy_regular['How often do you buy sex toys for yourself?'] == 'bought once or twice'
            ])) / len(df_flattened[df_flattened['How often do you buy sex toys for yourself?'] == 'bought once or twice'])
        )
    print(
        df_no_need_to_buy_regular['How often do you buy sex toys for yourself?'].value_counts(normalize=True))
    df_no_need_to_buy_regular = df_no_need_to_buy_regular[
        df_no_need_to_buy_regular['How often do you buy sex toys for yourself?'] == 'bought once or twice'
    ]
    for feature in ['sexuality_cleaned', 'gender_cleaned', 'partner_cleaned']:
        print(
            df_no_need_to_buy_regular[feature].value_counts(normalize=True))


df_no_discomfort = df_flattened[
    df_flattened['I feel discomfort when buying sex toys'] == 1
]

def print_sources_of_discomfort(df_no_discomfort):
    """
    Print demographic and sources of discomfort distributions for those
    who answered that they have no discomfort
    """
    print(
        'Size',
        len(df_no_discomfort),
        float(len(df_no_discomfort)) / len(df_flattened))
    for feature in ['sexuality_cleaned', 'gender_cleaned', 'partner_cleaned']:
        print(
            df_no_discomfort[feature].value_counts(normalize=True))
    print_personas_stats(df_no_discomfort, df_no_discomfort)


df_frequent_buyers = df_flattened[
    df_flattened['How often do you buy sex toys for yourself?'] == 'buying regularly (over 3 times a year)']
df_female_frequent_buyers = df_frequent_buyers[
    df_frequent_buyers['gender_cleaned'] == 'female']
df_male_frequent_buyers = df_frequent_buyers[
    df_frequent_buyers['gender_cleaned'] == 'male']

def print_frequent_buyers(df_, df):
    """
    Print demographics and usage distributions for frequent buyers
    """
    print(
        'Size',
        len(df_),
        float(len(df_)) / len(df))
    for feature in ['sexuality_cleaned', 'partner_cleaned']:
        print(
            df_[feature].value_counts(normalize=True))
    print_personas_stats(df_, df_)


df_satisfied = df_flattened[df_flattened.not_buy_reasons == 'no_need_satisfied']
def print_satisfied_people(df_, df):
    """
    Print demographics and usage distributions for users
    who names "being satsifed" as the main reason not to buy
    """
    print(
        'Size',
        len(df_),
        float(len(df_)) / len(df))
    for feature in ['sexuality_cleaned', 'gender_cleaned', 'partner_cleaned']:
        print(
            df_[feature].value_counts(normalize=True))


## Find correlations
def _get_new_column_name(feature, value):
    """
    Generate new column name after feature binarisation
    """
    return '--'.join([
            feature,
            value,
        ]).lower().replace(' ', '_')


def _get_possible_values(series):
    """
    Get all possible values for multi choices column
    """
    res = {}
    for choices in series.unique():
        if not choices or is_nan(choices): continue
        for choice in choices.split(','):
            choice = choice.lower().strip()
            if not choice: continue
            res[choice] = 1
    return list(res.keys())

def get_binarised_columns(
        df, 
        simple_categorical_features, 
        multiple_categorical_features,
        numeric_features):
    """
    Generate new dataset with binarised categorical columns
    """
    df = df.copy()
    final_columns = []
    for feature in simple_categorical_features:
        for value in df[feature].unique():
            if not value or is_nan(value): continue
            new_column = _get_new_column_name(feature, value)
            df[new_column] = df.apply(
                lambda x: x[feature] == value,
                axis=1)
            final_columns.append(new_column)

    for feature in multiple_categorical_features:
        for value in _get_possible_values(df[feature]):
            if not value or is_nan(value): continue
            new_column = _get_new_column_name(feature, value)
            df[new_column] = df.apply(
                lambda x: value in x[feature] if x[feature] and not is_nan(x[feature]) else False,
                axis=1)
            final_columns.append(new_column)

    for feature in numeric_features:
        new_column = feature.strip().lower().replace(' ', '_')
        df[new_column] = df[feature]
        final_columns.append(new_column)
    return df, final_columns


df_bin, final_columns = get_binarised_columns(
        df, 
        simple_categorical_features, 
        multiple_categorical_features,
        numeric_features)

result_columns = [
    column 
    for column in final_columns
    if any(
        val in column
        for val in [
            'how_often_do_you_buy_sex_toys_for_yourself?',
            'sources_of_discomfort',
            'not_buy_reasons',
            'i_use_sex_toys_regularly_in_my_sex_live', 
            'i_use_sex_toys_with_my_partners', "i'm_adventurous_when_it_comes_to_trying_new_things", 
            'i_feel_discomfort_when_buying_sex_toys']
        )
]

def find_correlations(df_bin, final_columns, result_columns):
    """
    Find correlatios between usage partners and demographic features
    """
    df_bin = df_bin[final_columns]
    df_bin = df_bin.fillna(0)
    for column in [
            'i_use_sex_toys_regularly_in_my_sex_live', 
            'i_use_sex_toys_with_my_partners',
            "i'm_adventurous_when_it_comes_to_trying_new_things", 
            'i_feel_discomfort_when_buying_sex_toys']:
        df_bin[column] = df_bin.apply(
            lambda x: float(x[column]) / df_bin[column].max(),
            axis=1)
    for column in result_columns:
        find_correlation(
            df_bin, column, final_columns)
        raw_input()


def find_correlation(df_bin, column, final_columns):
    """
    Find correlation with given feature
    Use xgboost to get most important features
    """
    import xgboost

    def _get_correlation(df_bin, feature, column):
        if feature in [
            'i_use_sex_toys_regularly_in_my_sex_live', 
            'i_use_sex_toys_with_my_partners',
            "i'm_adventurous_when_it_comes_to_trying_new_things", 
            'i_feel_discomfort_when_buying_sex_toys']:
            df_bin_ = df_bin[df_bin[column] == True]
            return (
                    df_bin_[feature].mean(),
                    df_bin[feature].corr(df_bin[column]),
                )
        return df_bin[feature].corr(df_bin[column])

    print('Find correlations: ', column)
    clf = xgboost.XGBClassifier(n_estimators=300, max_depth=5)
    x_columns_list = list(set(final_columns) - set([column, ]))
    X = df_bin[x_columns_list].as_matrix()
    y = df_bin[[column, ]].as_matrix()
    clf.fit(X, y)
    feature_importances = zip(x_columns_list, clf.feature_importances_)
    most_important_features = sorted(
        filter(
            lambda x: x[-1] > 0,
            feature_importances),
        key=lambda x: x[-1],
        reverse=True)[:5]
    for feature in most_important_features:
        print(
            column, '<-', feature[0], feature[1], 
            _get_correlation(df_bin, feature[0], column))
    print
