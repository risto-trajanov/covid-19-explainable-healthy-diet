import numpy as np
import os
import pandas as pd
from sklearn import preprocessing
import settings

output_folder = settings.output_folder
data_folder = settings.raw_data_folder
features_folder = settings.features_folder

food_df = pd.read_csv(f'{output_folder}diet.csv')
icd_df = pd.read_csv(f'{output_folder}icd_10.csv')
icd_scaled_df = pd.read_csv(f'{output_folder}icd_10_scaled.csv')
countries_df = pd.read_csv(f'{data_folder}control/countries.csv')
control_df = pd.read_csv(f'{data_folder}control/country_development_data.csv')
targets_df = pd.read_csv(f'{settings.raw_data_folder}control/regression_targets.csv')

food_features = pd.read_csv(f'{features_folder}selected_features_rfe_shap_food.csv')
icd_features = pd.read_csv(f'{features_folder}selected_features_rfe_shap_comorbidity.csv')
control_features = pd.read_csv(f'{features_folder}selected_features_rfe_shap_development.csv')


def save_df(df, name):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    df.to_csv(output_folder + name)


def get_selected(df, selected_features):
    if 'Country' in df.columns:
        df_countries = df['Country']
        df = df[selected_features.tolist()]
        df['Country'] = df_countries
        return df
    return df[selected_features.tolist()]


def main():
    icd = get_selected(icd_df, icd_features['selected_features'])
    icd_scaled = get_selected(icd_scaled_df, icd_features['selected_features'])
    food = get_selected(food_df, food_features['selected_features'])
    control = get_selected(control_df, control_features['selected_features'])

    save_df(icd, 'icd_selected_features.csv')
    save_df(icd_scaled, 'icd_scaled_selected_features.csv')
    save_df(food, 'food_selected_features.csv')
    save_df(control, 'control_selected_features.csv')

    icd_food = pd.merge(icd, food, on='Country')
    save_df(icd_food, 'icd_food_selected_features.csv')

    icd_scaled_food = pd.merge(icd_scaled, food, on='Country')
    save_df(icd_scaled_food, 'icd_scaled_food_selected_features.csv')

    icd_control = pd.merge(icd, control, on='Country')
    save_df(icd_control, 'icd_control_selected_features.csv')

    icd_scaled_control = pd.merge(icd_scaled, control, on='Country')
    save_df(icd_scaled_control, 'icd_scaled_control_selected_features.csv')

    food_control = pd.merge(food, control, on='Country')
    save_df(food_control, 'food_control_selected_features.csv')

    icd_food_control = pd.merge(icd_food, control, on='Country')
    save_df(icd_food_control, 'icd_food_control_selected_features.csv')

    icd_food_control = pd.merge(icd_scaled_food, control, on='Country')
    save_df(icd_food_control, 'icd_scaled_food_control_selected_features.csv')


if __name__ == '__main__':
    main()
