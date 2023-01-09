import numpy as np
import pandas as pd
from datetime import datetime
from dask import dataframe

def prepare_train(data, feature_path: str):
    """Подготовка данных для работы модели"""

    # загрузка данных признаков"
    feats = dataframe.read_csv(feature_path, sep='\t')
    feats = feats.drop(columns='Unnamed: 0')

    # преобразование времени покупки в timestamp"
    data['buy_time'] = data.buy_time.apply(lambda x: datetime.fromtimestamp(x))
    feats['buy_time'] = feats.buy_time.apply(lambda x: datetime.fromtimestamp(x), meta=('buy_time', 'datetime64[ns]'))

    # объединение данных с признаками по id клиента
    df = feats.merge(data, on='id').compute()

    # удаление дубликатов в данных по большей разнице между временем покупки в данных и временем покупки в feats
    df['diff_time'] = -np.abs(df['buy_time_x'] - df['buy_time_y'])
    df['diff_time'] = df.diff_time.apply(lambda x: x.days)  # преобразование разницы времени в числовое значение
    df.sort_values('diff_time', ascending=False, inplace=True)
    df.drop_duplicates(['id', 'buy_time_y', 'vas_id'], inplace=True)

    df.set_index('id', inplace=True)

    # дополнительные признаки
    df['buy_month_feat'] = df.buy_time_x.apply(lambda x: x.month)
    df['buy_month'] = df.buy_time_y.apply(lambda x: x.month) # месяц покупки из выборки
    df['buy_day'] = df.buy_time_y.apply(lambda x: x.day) # день покупки из выборки

    # преобразование в категориальный признак
    df.loc[df['252'] > 1, '252'] = 2
    
    # удаление из выборки 19 ноября
    df = df.loc[~((df['buy_month']==11) & (df['buy_day']==19))]

    # удаление ненужных признаков
    df.drop(['buy_time_y', 'buy_time_x', 'buy_day'], axis=1, inplace=True)

    return df


def prepare_test(data, feature_path: str):
    """Подготовка данных для работы модели"""

    # загрузка данных признаков"
    feats = dataframe.read_csv(feature_path, sep='\t')
    feats = feats.drop(columns='Unnamed: 0')

    # преобразование времени покупки в timestamp"
    data['buy_time'] = data.buy_time.apply(lambda x: datetime.fromtimestamp(x))
    feats['buy_time'] = feats.buy_time.apply(lambda x: datetime.fromtimestamp(x), meta=('buy_time', 'datetime64[ns]'))

    # объединение данных с признаками по id клиента
    df = feats.merge(data, on='id').compute()

    # удаление дубликатов в данных по большей разнице между временем покупки в данных и временем покупки в feats
    df['diff_time'] = -np.abs(df['buy_time_x'] - df['buy_time_y'])
    df['diff_time'] = df.diff_time.apply(lambda x: x.days)  # преобразование разницы времени в числовое значение
    df.sort_values('diff_time', ascending=False, inplace=True)
    df.drop_duplicates(['id', 'buy_time_y', 'vas_id'], inplace=True)

    df.set_index('id', inplace=True)

    # дополнительные признаки
    df['buy_month_feat'] = df.buy_time_x.apply(lambda x: x.month)

    # преобразование в категориальный признак
    df.loc[df['252'] > 1, '252'] = 2

    # удаление ненужных признаков
    df.drop(['buy_time_y', 'buy_time_x'], axis=1, inplace=True)

    return df
