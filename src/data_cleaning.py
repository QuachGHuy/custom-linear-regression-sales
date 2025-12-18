import pandas as pd
def clean_data(df):
    """
    Takes a raw DataFrame and returns a cleaned DataFrame.
    """
    # 1. Remove duplicates
    df = df.drop_duplicates()

    # 2. Remove outliers
    # Remove unreasonable negative values.
    cols_to_check = ["Radio", "TV", "Newspaper", "Sales"]
    for col in cols_to_check:
        if col in df.columns:
            df = df[df[col] >= 0]

    return df