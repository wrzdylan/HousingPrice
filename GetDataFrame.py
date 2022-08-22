"""Module GetDataFrame"""

# Import libraries
import pandas as pd
import numpy as np
import warnings
from typing import List
from scipy.stats import skew, boxcox_normmax
from scipy.special import boxcox1p
from sklearn.preprocessing import OrdinalEncoder

warnings.filterwarnings("ignore")


class GetDataFrame:
    def __init__(self, df_name="train"):
        self.df: pd.DataFrame = pd.read_csv(f"dataset/{df_name}.csv")
        self.numerical_features: List[str] = list()
        self.categorical_features: List[str] = list()
        self.feature_outlier_count = {}

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

        # --- Fill NA values ---
        self.__imputing_missing_values()

        # --- Add features ---
        self.__add_features()

        # --- Clean outliers ---
        self.fix_outliers()

        # --- Transform variables types ---
        self.__numeric_vars_to_categorical()

        # --- fix skewed numeric features ---
        self.transform_skewed_features()

        # --- Ordinal variables ---
        self.__ordinal_encode()

        # --- dummy categorical features
        self.df = pd.get_dummies(self.df)

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

        self.df['TotalSqrFootage'] = (
                self.df['BsmtFinSF1'] + self.df['BsmtFinSF2'] +
                self.df['1stFlrSF'] + self.df['2ndFlrSF']
        )

        self.df['YrBltAndRemod'] = self.df['YearBuilt'] + self.df['YearRemodAdd']

        self.df["LivLotRatio"] = self.df.GrLivArea / self.df.LotArea

        self.df["Spaciousness"] = self.df["TotalFlrSFAbvGrd"] / self.df.TotRmsAbvGrd

        # Adding all the bathrooms
        self.df["TotalBath"] = (
                self.df['FullBath'] + (0.5 * self.df['HalfBath']) +
                self.df['BsmtFullBath'] + (0.5 * self.df['BsmtHalfBath'])
        )

        # Adding square feet of all Porch
        self.df["TotalPorchSF"] = self.df[
            ["OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch", "WoodDeckSF"]
        ].sum(axis=1)

        self.df['has_pool'] = self.df['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
        self.df['has_2nd_floor'] = self.df['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
        self.df['has_garage'] = self.df['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
        self.df['has_bsmt'] = self.df['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
        self.df['has_fireplace'] = self.df['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)

    def __numeric_vars_to_categorical(self) -> None:
        """ Transform MSSubClass, OverallCond, YrSold, MoSold into categorical because
        in the data description, they are linked to a nominal value.

        :return:
        """
        cols_to_categorical = ["MSSubClass", "YrSold", "MoSold"]
        self.df[cols_to_categorical] = self.df[cols_to_categorical].astype(str)

        # adding MSSubClass to categorical Feature list
        self.categorical_features.extend(cols_to_categorical)
        # removing it from numerical feature list
        self.numerical_features = [i for i in self.numerical_features if i not in cols_to_categorical]

    def __ordinal_encode(self) -> None:
        """ Convert nominal features into ordinal features, mandatory for some models

        :return:
        """
        five_levels = ["Po", "Fa", "TA", "Gd", "Ex"]

        ordinal_mappings = {
            "ExterQual": five_levels,
            "LotShape": ['Reg', 'IR1', 'IR2', 'IR3'],
            "BsmtQual": five_levels,
            "BsmtCond": five_levels,
            "BsmtExposure": ['No', 'Mn', 'Av', 'Gd'],
            "BsmtFinType1": ['Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'],
            "BsmtFinType2": ['Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'],
            "HeatingQC": five_levels,
            "Functional": ['Sev', 'Maj2', 'Maj1', 'Mod', 'Min2', 'Min1', 'Typ'],
            "FireplaceQu": five_levels,
            "KitchenQual": five_levels,
            "GarageFinish": ['Unf', 'RFn', 'Fin'],
            "GarageQual": five_levels,
            "GarageCond": five_levels,
            "PoolQC": five_levels[1:],
            "Fence": ['MnWw', 'GdWo', 'MnPrv', 'GdPrv'],
            "LandSlope": ['Sev', 'Mod', 'Gtl'],
            "PavedDrive": ['N', 'P', 'Y'],
            "CentralAir": ['N', 'Y'],
        }
        ordinal_mappings = {key: ["None"] + value for key, value in ordinal_mappings.items()}

        encoder = OrdinalEncoder(
            categories=list(ordinal_mappings.values()),
            handle_unknown='use_encoded_value',
            unknown_value=np.nan
        )
        self.df.loc[:, ordinal_mappings.keys()] = encoder.fit_transform(self.df.loc[:, ordinal_mappings.keys()])

    def transform_skewed_features(self) -> None:
        """ Use scipy function boxcox1p to normalize skewed features

        :return:
        """
        skewed_feats = self.df[self.numerical_features].apply(
            lambda x: skew(x.dropna())
        ).sort_values(ascending=False)

        high_skew = skewed_feats[skewed_feats > 0.75]
        skew_index = high_skew.index

        for i in skew_index:
            self.df[i] = boxcox1p(self.df[i], boxcox_normmax(self.df[i] + 1))

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
        # Can use z-score or inter quartile range
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
