import pandas as pd
import numpy as np
import os
from sklearn import preprocessing
import settings

output_folder = settings.output_folder
data_folder = f'{settings.raw_data_folder}comorbidity/'
country_code_file = 'country_codes.csv'
country_codes = data_folder + country_code_file

# A00-B99	Certain infectious and parasitic diseases
# C00-D48	Neoplasms
# E00-E88	Endocrine, nutritional and metabolic diseases
# F01-F99	Mental and behavioural disorders
# G00-G98	Diseases of the nervous system
# H00-H57	Diseases of the eye and adnexa
# H60-H93	Diseases of the ear and mastoid process
# I00-I99	Diseases of the circulatory system
# J00-J98	Diseases of the respiratory system
# K00-K92	Diseases of the digestive system
# L00-L98	Diseases of the skin and subcutaneous tissue
# M00-M99	Diseases of the musculoskeletal system and connective tissue
# N00-N98	Diseases of the genitourinary system
# O00-O99	Pregnancy, childbirth and the puerperium
# P00-P96	Certain conditions originating in the perinatal period
# Q00-Q99	Congenital malformations, deformations and chromosomal abnormalities
# R00-R99	Symptoms, signs and abnormal clinical and laboratory findings, not elsewhere classified
# V01-Y89	External causes of morbidity and mortality

death_cause_map = {
    'A': 'infectious and parasitic diseases',
    'B': 'infectious and parasitic diseases',
    'C': 'Neoplasms',
    'D': 'Neoplasms',
    'E': 'Endocrine, nutritional and metabolic diseases',
    'F': 'Mental and behavioural disorders',
    'G': 'Diseases of the nervous system',
    'H': 'Diseases of the eye and adnexa and ear and mastoid process',
    'I': 'Diseases of the circulatory system',
    'J': 'Diseases of the respiratory system',
    'K': 'Diseases of the digestive system',
    'L': 'Diseases of the skin and subcutaneous tissue',
    'M': 'Diseases of the musculoskeletal system and connective tissue',
    'N': 'Diseases of the genitourinary system',
    'O': 'Pregnancy, childbirth and the puerperium',
    'P': 'Certain conditions originating in the perinatal period',
    'Q': 'Congenital malformations, deformations and chromosomal abnormalities',
    'R': 'Symptoms, signs and abnormal clinical and laboratory findings, not elsewhere classified',
    'V': 'External causes of morbidity and mortality'
}

part_year = {
    4: ['2013', '2014', '2015', '2016'],
    5: ['2017', '2018']
}


def save_df(df, name):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    df.to_csv(output_folder + name)


def scale_df(df):
    x = df.values  # returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df_scaled = pd.DataFrame(x_scaled)
    return df_scaled


def average_icd(list_dfs):
    concat_icd_dfs = pd.concat(list_dfs).groupby(level=0).mean()
    # drop all columns with 80% nans
    thresh = len(concat_icd_dfs) * .8
    concat_icd_dfs_dropped_nans = concat_icd_dfs.dropna(thresh=thresh, axis=1)
    return concat_icd_dfs_dropped_nans


def rename_columns(df, year, columns):
    columns = columns
    for column in columns:
        new_column_name = column + '_' + year
        df = df.rename({column: new_column_name}, axis=1)
    return df


def pivot_tables(df):
    pivot_1 = pd.pivot_table(df, values='Deaths1', index=['Country', 'Cause'], aggfunc=np.sum)
    pivot_1_df = pivot_1.reset_index()
    pivot_1_df_pivot_2 = pd.pivot_table(pivot_1_df, values='Deaths1', index=['Country'], columns=['Cause'])
    pivot_1_df_pivot_2_df = pivot_1_df_pivot_2.reset_index()
    return pivot_1_df_pivot_2_df


def iter_files():
    country_codes_df = pd.read_csv(country_codes)
    all_years_icd = []
    for i in range(4, 6):
        mort = pd.read_csv(data_folder + f'Morticd10_part{i}.csv', usecols=[0, 3, 5, 6, 9])
        years = part_year.get(i)
        for year in years:
            mort_year = mort[mort['Year'] == int(year)]
            mort_pivot = pivot_tables(mort_year)
            mort_pivot = mort_pivot.set_index('Country')
            all_years_icd.append(mort_pivot)

    concat_icd_dfs = pd.concat(all_years_icd).groupby(level=0)
    concat_icd_dfs_mean = pd.concat(all_years_icd).groupby(level=0).mean()

    list_keys = list(death_cause_map.keys())

    concat_icd_dfs_mean = concat_icd_dfs_mean.apply(lambda x: x)
    new_df = pd.DataFrame()
    for char in list_keys:
        cause = [col for col in concat_icd_dfs_mean.columns if col[0] == char]
        new_df[char] = concat_icd_dfs_mean[cause].sum(axis=1)

    mean_a_b = new_df[['A', 'B']].sum(axis=1)
    mean_c_d = new_df[['C', 'D']].sum(axis=1)

    new_df_drop = new_df.drop(['A', 'B', 'C', 'D'], axis=1)
    new_df_drop['A'] = mean_a_b
    new_df_drop['C'] = mean_c_d

    for column in new_df_drop.columns:
        new_df_drop = new_df_drop.rename(columns={column: death_cause_map.get(column)})

    scaled = scale_df(new_df_drop)
    scaled = scaled.apply(lambda x: x)
    new_df_drop['country'] = new_df_drop.index
    merged = pd.merge(new_df_drop, country_codes_df, on='country')
    merged = merged.drop(['country'], axis=1)
    merged = merged.rename({'name': 'Country'}, axis=1)
    save_df(merged, 'icd_10.csv')
    scaled['Country'] = merged['Country']
    scaled.columns = merged.columns
    save_df(scaled, 'icd_10_scaled.csv')


def main():
    iter_files()


if __name__ == '__main__':
    main()
