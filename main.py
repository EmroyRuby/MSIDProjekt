import pandas as pd
import statsmodels.api as sm
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.metrics import r2_score
from pandas_profiling import ProfileReport


# import_df function imports the abalone.data dataset and adds names of the columns (in original data they're missing)
def import_df() -> pd.DataFrame:
    df_raw = pd.read_csv("abalone.data")
    names = ['Sex', 'Length', 'Diameter', 'Height', 'Weight', 'Shucked weight', 'Viscera weight', 'Shell weight',
             'Rings']
    df_raw.columns = names
    return df_raw


# generates a pandas profiling report of the given DataFrame and saves with the given name
def gen_raport(df_to_prof: pd.DataFrame, out_file: str) -> None:
    prof = ProfileReport(df_to_prof)
    prof.to_file(output_file=out_file)


# pre_processing of data regarding Sex of Abalone, changes M, I and F to a corresponding 01, 10, 00
def sex_to_bin(df: pd.DataFrame) -> pd.DataFrame:
    status = pd.get_dummies(df['Sex'], drop_first=True)
    df = pd.concat([df, status], axis=1)
    df.drop(['Sex'], axis=1, inplace=True)
    return df


def print_distribution_of_data(df: pd.DataFrame) -> None:
    for name in df.columns:
        sns.displot(df[name])
        plt.show(block=True)


# splitting the DataFrame into train and test sets in a 7:3 ratio, without randomness
def split_train_test(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    np.random.seed(0)
    df_train, df_test = train_test_split(df, train_size=0.7, test_size=0.3, random_state=100)
    y_train = df_train.pop('Rings')
    x_train = df_train
    y_test = df_test.pop('Rings')
    x_test = df_test
    return x_train, y_train, x_test, y_test


# calculate RFE of the given DataFrame model and return a list containing:
# name of variable, bool of being essential to the model and corresponding RFE ranking
def calculate_rfe(x_train: pd.DataFrame, y_train: pd.DataFrame) -> list:
    lm = LinearRegression()
    lm.fit(x_train, y_train)
    rfe = RFE(lm)
    rfe = rfe.fit(x_train, y_train)
    res = list(zip(x_train.columns, rfe.support_, rfe.ranking_))
    return res


# drop all non-essential columns from the DataFrame
def drop_columns(x_train: pd.DataFrame, x_test: pd.DataFrame, res: list) -> tuple[pd.DataFrame, pd.DataFrame]:
    col = [x[0] for x in res if x[1]]
    return x_train[col], x_test[col]


# leave only essential columns and with the ranking score of 2 or 3
def drop_columns_v2(x_train: pd.DataFrame, x_test: pd.DataFrame, res: list) -> tuple[pd.DataFrame, pd.DataFrame]:
    col = [x[0] for x in res if x[1] or x[2] == 2 or x[2] == 3]
    return x_train[col], x_test[col]


# build the linear model of our data
def build(x_test_rfe: pd.DataFrame, y_test: pd.DataFrame, x_train_rfe: pd.DataFrame, y_train: pd.DataFrame):
    # adding constant
    x_train_rfe = sm.add_constant(x_train_rfe)
    x_test_rfe = sm.add_constant(x_test_rfe)
    # building the model using OLS method
    lm = sm.OLS(y_train, x_train_rfe).fit()
    # prediction of rings based on the model
    y_train_rings = lm.predict(x_train_rfe)
    print(lm.summary())
    # display of error distribution of the model
    sns.displot((y_train - y_train_rings))
    plt.xlabel('Errors')
    plt.show(block=True)
    # predicting the values of test DataFrame
    y_test_rings = lm.predict(x_test_rfe)
    # printing the corresponding r2_scores
    print(f'train data score: {r2_score(y_true=y_train, y_pred=y_train_rings)}')
    print(f'test data score: {r2_score(y_true=y_test, y_pred=y_test_rings)}')


if __name__ == '__main__':
    df = import_df()
    out_file = 'Report.html'
    # generating a report can take a while, feel free to skip this part (will generate only once,
    # until deleted or name changes)
    if not os.path.isfile(out_file):
        gen_raport(df, out_file)
    print_distribution_of_data(df)
    df = sex_to_bin(df)
    x_train, y_train, x_test, y_test = split_train_test(df)
    res = calculate_rfe(x_train, y_train)
    # uncomment the line with the version of drop_columns you want to use, comment the other one
    # x_train_rfe, x_test_rfe = drop_columns(x_train, x_test, res)
    x_train_rfe, x_test_rfe = drop_columns_v2(x_train, x_test, res)
    build(x_test_rfe, y_test, x_train_rfe, y_train)
