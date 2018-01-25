import math

import pandas as pd

def is_nan(val):
    return type(val) == float and math.isnan(val)

def flatten_multiple_choices(
        df, feature):
    """
    Dublicate rows with multiple choices in "feature" column
    Transform row with N choices to N rows with each choice in "feature" column
    """
    df_for_append = pd.DataFrame()
    for index, row in df.iterrows():
        if not row[feature] or is_nan(row[feature]):
            continue
        values = row[feature].split(',')
        df.at[index, feature] = values[0]
        for value in values[1:]:
            if not value: continue
            new_row = row.copy()
            new_row[feature] = value
            df_for_append.append(new_row)
    df = df.append(df_for_append, ignore_index=True)
    return df
