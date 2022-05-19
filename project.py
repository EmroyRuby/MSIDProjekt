import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from pandas_profiling import ProfileReport
import statsmodels.api as sm
import os


def import_df() -> pd.DataFrame:
    df_raw = pd.read_csv("abalone.data")
    names = ['Sex', 'Length', 'Diameter', 'Height', 'Weight', 'Shucked weight', 'Viscera weight', 'Shell weight',
             'Rings']
    df_raw.columns = names
    return df_raw


def gen_raport(df_to_prof: pd.DataFrame, out_file: str) -> None:
    prof = ProfileReport(df_to_prof)
    prof.to_file(output_file=out_file)


def sex_to_bin(x: str) -> int:
    if x == 'M':
        return 1
    return 0


def linear_regresion(x_train: pd.DataFrame, y_train: pd.DataFrame) -> list:
    lm = LinearRegression()
    lm.fit(x_train, y_train)
    rfe = RFE(lm)
    rfe = rfe.fit(x_train, y_train)
    res = list(zip(x_train.columns, rfe.support_, rfe.ranking_))
    return res


def build(x_test: pd.DataFrame, y_test: pd.DataFrame, x_train: pd.DataFrame, y_train: pd.DataFrame, selected_col: list):
    x_train_rfe = x_train[selected_col]
    x_train_rfe = sm.add_constant(x_train_rfe)
    lm = sm.OLS(y_train, x_train_rfe).fit()

    print(lm.summary())


if __name__ == '__main__':
    df = import_df()
    out_file = 'output.html'
    # generating a raport can take a while, feel free to skip this part
    if not os.path.isfile(out_file):
        gen_raport(df, out_file)
    df['Sex'] = df['Sex'].apply(sex_to_bin)
    df_train, df_test = train_test_split(df)
    y_train = df_train.pop('Rings')
    x_train = df_train
    y_test = df_test.pop('Rings')
    x_test = df_test
    col = [x[0] for x in linear_regresion(x_train, y_train) if x[1]]
    build(x_test, y_test, x_train, y_train, col)
