import pandas as pd
import numpy as np
import os
from sklearn import preprocessing
import settings

files = ['Fat_Supply_Quantity_Data.csv', 'Food_Supply_kcal_Data.csv', 'Food_Supply_Quantity_kg_Data.csv']
output_folder = settings.output_folder
data_folder = f'{settings.raw_data_folder}diet/'

if __name__ == '__main__':
    df_1 = pd.read_csv(data_folder + files[0])
    df_2 = pd.read_csv(data_folder + files[1])
    df_3 = pd.read_csv(data_folder + files[2])
    countries = pd.read_csv(data_folder + 'countries.csv')

    df_1.drop(['Confirmed', 'Deaths', 'Recovered'], axis=1)
    df_2.drop(['Confirmed', 'Deaths', 'Recovered'], axis=1)
    df_3.drop(['Confirmed', 'Deaths', 'Recovered'], axis=1)

    df_1.drop(['Confirmed', 'Deaths', 'Recovered'], axis=1)
    df_2.drop(['Confirmed', 'Deaths', 'Recovered'], axis=1)
    df_3.drop(['Confirmed', 'Deaths', 'Recovered'], axis=1)

    list_dfs = [df_1, df_2, df_3]

    merged_df = pd.concat([df_1, df_2, df_3]).groupby(level=0).mean()

    merged_df['Country'] = countries['Country']

    merged_df.to_csv(f'{output_folder}diet.csv')
