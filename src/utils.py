import requests
from category_encoders.target_encoder import TargetEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle


def download_dataset(destination):
    url = "https://docs.google.com/uc?export=download"
    file_id = '15hj_kpNICz-Czl7hpDjOjxKlU6bElCRr'

    session = requests.Session()

    response = session.get(url, params={'id': file_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(url, params=params, stream=True)

    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None


def save_response_content(response, destination):
    chunk_size = 32768
    with open(destination, "wb") as f:
        for chunk in response.iter_content(chunk_size):
            if chunk:
                f.write(chunk)


def train_test_split_time(dataframe, test_n_days=5):
    dates = sorted(dataframe['current_dt'].unique())
    train = dataframe[dataframe['current_dt'].isin(dates[:-test_n_days])]
    val = dataframe[dataframe['current_dt'].isin(dates[-test_n_days:])]
    return train, val


def change_dtypes(dataframe, categorical_cols):
    df = dataframe.copy()
    for col in categorical_cols:
        df[col] = df[col].astype('object')

    return df


def split_dataframe(dataframe, on_time):
    df = dataframe.copy()
    if on_time:
        train, val = train_test_split_time(df)
    else:
        train, val = train_test_split(df, test_size=0.2, random_state=42)
    return train, val


def fill_nans(dataframe):
    card_type_v = 3
    dataframe['card_type'] = dataframe['card_type'].fillna(card_type_v)
    return dataframe


def merge_categories_custom(dataframe, col, subset):
    df = dataframe.copy()

    if subset == 'train':

        counts_df = df.groupby([col, 'is_success'])['index'].count().unstack()
        percentes = counts_df.T.div(counts_df.T.sum()).T
        percentes = percentes.reset_index().fillna(0)
        percentes.columns = [col, f'not_success_{col}', f'success_{col}']
        percentes = percentes.drop([f'not_success_{col}'], axis=1)
        percentes[f'success_{col}'] = percentes[f'success_{col}'].apply(lambda x: round(x, 3))
        percentes.to_csv(f'data/percentes_{col}.csv', index=False)
    else:
        percentes = pd.read_csv(f'data/percentes_{col}.csv')

    df_prep = pd.merge(df, percentes, on=col, how='left')
    df_prep[col] = df_prep[f'success_{col}'].apply(lambda x: f'{col}_' + str(x))
    df_prep.drop([f'success_{col}'], axis=1, inplace=True)

    return df_prep


def reduce_cardinality(dataframe, categorical_cols, subset):
    df = dataframe.copy()
    df = df.reset_index()
    for col in categorical_cols:
        df = merge_categories_custom(df, col, subset)

    df.drop(['index'], axis=1, inplace=True)
    return df


def encode_categorical(df, categorical_cols, subset):
    df['gender'] = df.gender.map(lambda x: 0 if x == 'm' else 1)

    if subset == 'train':
        encoder = TargetEncoder()
        df[categorical_cols] = encoder.fit_transform(df[categorical_cols], df['is_success'])
        pickle.dump(encoder, open('enc.pkl', 'wb'))
    else:
        encoder = pickle.load(open('enc.pkl', 'rb'))
        df[categorical_cols] = encoder.transform(df[categorical_cols])

    return df


def preprocess(dataframe, categorical_cols, subset='train'):
    df = dataframe.copy()
    df = change_dtypes(df, categorical_cols)

    if subset == 'train':
        df = df.drop_duplicates()
        df = fill_nans(df)
        df = reduce_cardinality(df, categorical_cols, subset)
        df = encode_categorical(df, categorical_cols, subset)

    else:
        df = fill_nans(df)
        df = reduce_cardinality(df, categorical_cols, subset)
        df = encode_categorical(df, categorical_cols, subset)

    return df
