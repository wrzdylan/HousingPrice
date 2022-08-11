"""Module GetDataFrame"""

# Import libraries
import pandas as pd
import numpy as np
import datetime
import warnings
from typing import List

warnings.filterwarnings("ignore")


class GetDataFrame:
    def __init__(self, df_name="train"):
        self.categorical_columns = None
        self.cat_feature = None
        self.tm = None
        self.df: pd.DataFrame = pd.read_csv(f"dataset/{df_name}.csv")
        self.numerical_features: List[str] = list()
        self.categorical_features: List[str] = list()
        self.feature_outlier_count = {}
        self.house_price = {}

    def get_cleaned_df(self):
        # Unnecessary for  the prediction process
        self.df.drop("Id", axis=1, inplace=True)
        # All records are "AllPub"
        self.df.drop("Utilities", axis=1, inplace=True)

        # SalePrice is skewed then we use log because the min value is above 30k, no need of log1p
        self.df["SalePrice"] = np.log(self.df["SalePrice"])

        # create a list of numerical features
        self.numerical_features = list(self.df.select_dtypes(include=[np.number]).columns.values)
        # create a list of features that are categorical
        self.categorical_features = list(self.df.select_dtypes(include=[np.object]).columns.values)

        # --- Clean outliers ---
        self.fix_outliers()

        # --- Fill NA values ---
        self.__imputing_missing_values()

        # --- Add features ---
        self.__add_features()

        # --- Transform variables types ---
        self.__numeric_vars_to_categorical()

        # ---> I'm HERE <---
        # --- Ordinal variables ---
        # https://machinelearningmastery.com/one-hot-encoding-for-categorical-data/
        # transforme les catégories en valeurs numérique -> One-Hot Encoding + dummy
        # Voir diff LabelEncoder et One-Hot encoding

        # created dummy variables for categorical features
        self.house_price = pd.concat(
            [self.df, pd.get_dummies(self.df[self.categorical_features], drop_first=True)], axis=1
        )

        # drop the actual categorical feature from list
        self.house_price.drop(columns=self.categorical_features, inplace=True)
        # reset index for the new dataframe
        self.house_price.reset_index(drop=True, inplace=True)

        # ---   TIME ---
        # A voir ce que c'est
        # lets create a constant time
        self.tm = datetime.time(10, 10)

        # convert the dateSold to unix timestamp
        self.house_price.dateSold = self.house_price.dateSold.apply(
            lambda x: datetime.datetime.combine(x, self.tm).timestamp()
        )

        # reassigning all the numerical features to the numerical_features variable as a list
        self.numerical_features = list(self.df.select_dtypes(include=[np.number]).columns.values)
        print(self.house_price.shape)

    def get_df(self):
        return self.df

    def __imputing_missing_values(self) -> None:
        """ Fill NA values
        Use description, mode, transform, None and 0 to fill missing data

        :return:
        """
        # transform instead of apply and group by with one index
        self.df["LotFrontage"] = self.df.groupby(["Neighborhood"])["LotFrontage"].transform(
            lambda x: x.fillna(x.median())
        )

        # date description says default value is Typical
        self.df["Functional"] = self.df["Functional"].fillna("Typ")

        # Columns to fill with the most common value
        na_cols_add_most_common = [
            "MSZoning",
            "Electrical",
            "SaleType",
            "KitchenQual",
            "Exterior1st",
            "Exterior2nd"
        ]
        # Categorical variables with few NA, fill with the most frequent value
        self.df[na_cols_add_most_common] = self.df[na_cols_add_most_common].fillna(
                self.df.mode().iloc[0]
        )

        # Example: NA value for Bsmt means no Bsmt, same for GarageYrBlt because no garage
        self.df[self.numerical_features] = self.df[self.numerical_features].fillna(0)

        # Example: data description for PoolQC says NA means no pool
        self.df[self.categorical_features] = self.df[self.categorical_features].fillna("None")

    def __add_features(self) -> None:
        """Help our models by creating new features

        :return:
        """
        # Total surface feature, very important to determine house prices  + add vectorization
        self.df["TotalFlrSFAbvGrd"] = self.df["1stFlrSF"] + self.df["2ndFlrSF"]
        self.df["TotalSF"] = self.df["TotalBsmtSF"] + self.df["TotalFlrSFAbvGrd"]

        self.df["LivLotRatio"] = self.df.GrLivArea / self.df.LotArea

        self.df["Spaciousness"] = self.df["TotalFlrSFAbvGrd"] / self.df.TotRmsAbvGrd

        # Adding all the bathrooms
        self.df["TotalBath"] = self.df[
            ["BsmtFullBath", "BsmtHalfBath", "FullBath", "HalfBath"]
        ].sum(axis=1)

        # Adding square feet of all Porch
        self.df["TotalPorchSF"] = self.df[
            ["OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch", "WoodDeckSF"]
        ].sum(axis=1)

    def __numeric_vars_to_categorical(self) -> None:
        """ Transform MSSubClass, OverallCond, YrSold, MoSold into categorical because
        in the data description, they are linked to a nominal value.

        :return:
        """
        cols_to_categorical = ["MSSubClass", "OverallCond", "YrSold", "MoSold"]
        self.df[cols_to_categorical] = self.df[cols_to_categorical].astype(str)

        # adding MSSubClass to categorical Feature list
        self.categorical_features.extend(cols_to_categorical)
        # removing it from numerical feature list
        self.numerical_features = [i for i in self.numerical_features if i not in cols_to_categorical]

    def get_columns_to_drop(self) -> List[str]:
        """Get the features that hold more than 90% of its data with a same value

        :return: List of columns name
        """
        drop_cols = [
            v
            for i, v in enumerate(self.df.columns)
            if ((self.df[v].value_counts().max() / len(self.df)) * 100) > 90
        ]
        return drop_cols

    def feature_outlier_make_count(self, to_print=False):
        # Can use z-score or interquartile range
        self.feature_outlier_count = {
            "1stFlrSF": 1,
            "BsmtFinSF1": 2,
            "BsmtFinSF2": 1,
            "EnclosedPorch": 2,
            "GarageArea": 4,
            "GrLivArea": 4,
            "LotArea": 7,
            "LotFrontage": 2,
            "MasVnrArea": 1,
            "OpenPorchSF": 3,
            "TotalBsmtSF": 4,
            "TotRmsAbvGrd": 1,
            "TotalSF": 2,
            "TotalPorchSF": 1,
            "WoodDeckSF": 3,
        }

        if to_print:
            for k, v in self.feature_outlier_count.items():
                if v:
                    print(
                        self.df.loc[
                            self.df[k].isin(sorted(self.df[k])[-v:]), [k, "SalePrice"]
                        ]
                    )

    def get_outliers(self, feature, index=-1):
        outlier = self.df.loc[
            self.df[feature] == sorted(self.df[feature])[index], [feature, "SalePrice"]
        ].sort_values(by=feature, ascending=False)
        return outlier

    def fix_outliers(self) -> None:
        """ Replace outliers by mean from 4 closest values

        :return:
        """
        self.feature_outlier_make_count()

        for k, v in self.feature_outlier_count.items():
            while v > 0:
                # replacing the outliers by taking mean of four closest feature value of the outlier
                # at the salePrice Range
                replace_with = self.df.loc[
                    (self.df["SalePrice"] - self.get_outliers(k)["SalePrice"].values[0])
                    .abs()
                    .argsort()[v: v + 4],
                    k
                ].mean()

                if (self.df[k].dtypes == np.int64) | (self.df[k].dtypes == np.int32):
                    self.df.loc[
                        self.df.index[self.get_outliers(k).index.values[0]], k
                    ] = int(replace_with)
                else:
                    self.df.loc[
                        self.df.index[self.get_outliers(k).index.values[0]], k
                    ] = round(replace_with, 1)

                v -= 1
                self.feature_outlier_count[k] = v
