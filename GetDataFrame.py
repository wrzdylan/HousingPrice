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
        self.numerical_features = self.df.select_dtypes(include=[np.number]).columns.values.tolist()
        # create a list of features that are categorical
        self.categorical_features = self.df.select_dtypes(include=[np.object]).columns.values.tolist()

        # --- Fill NA values ---
        self.__imputing_missing_values()

        # --- Add features ---
        self.__add_features()

        # --- Transform variables types ---
        self.fix_outliers()

        # Change some numerical variables to string, the values are the same we will use LabelEncoder after
        self.df.MSSubClass = self.df.MSSubClass.astype(str)
        self.df.OverallCond = self.df.OverallCond.astype(str)

        # --- Reduce the quantity of categorical values, is USEFUL ? For example, a near and adjacent are the same ---
        # A voir avec le RMSE si on gagne en performance
        # adding MSSubClass to categorical Feature list
        # self.categorical_features.append("MSSubClass")
        # removing it from numerical feature list
        # self.numerical_features.remove("MSSubClass")
        # Combine Categories that are not ordinal as ordinal categories (need refactorization)
        # self.df.BldgType.replace({"2fmCon": "Twnhs", "Duplex": "Twnhs"}, inplace=True)
        # self.df.BsmtExposure.replace({"Mn": "Av"}, inplace=True)
        # self.df.Condition1.replace(
        #     {"RRNn": "RRAn", "PosN": "PosA", "RRNe": "RRAe", "Feedr": "Artery"},
        #     inplace=True,
        # )
        # self.df.Exterior2nd.replace(
        #     {
        #         "MetalSd": "Wd Sdng",
        #         "Wd Shng": "Wd Sdng",
        #         "HbBoard": "Wd Sdng",
        #         "Plywood": "Wd Sdng",
        #         "Stucco": "Wd Sdng",
        #         "CBlock": "BrkFace",
        #         "Other": "BrkFace",
        #         "Stone": "BrkFace",
        #         "AsphShn": "BrkFace",
        #         "ImStucc": "BrkFace",
        #         "Brk Cmn": "BrkFace",
        #     },
        #     inplace=True,
        # )
        # self.df.Foundation.replace({"Wood": "Stone", "Slab": "Stone"}, inplace=True)
        # self.df.GarageType.replace(
        #     {
        #         "CarPort": "Detchd",
        #         "No Garage": "Detchd",
        #         "Basment": "Detchd",
        #         "2Types": "Detchd",
        #     },
        #     inplace=True,
        # )
        # self.df.LotShape.replace({"IR3": "IR2"}, inplace=True)
        # self.df.MSZoning.replace({"RH": "RM"}, inplace=True)
        # self.df.MasVnrType.replace(
        #     {"None": "BrkCmn", "Not present": "BrkCmn"}, inplace=True
        # )
        # self.df.Neighborhood.replace(
        #     {
        #         "BrDale": "MeadowV",
        #         "IDOTRR": "MeadowV",
        #         "NAmes": "Sawyer",
        #         "NPkVill": "Sawyer",
        #         "Mitchel": "Sawyer",
        #         "SWISU": "Sawyer",
        #         "Blueste": "Sawyer",
        #         "Blmngtn": "Gilbert",
        #         "SawyerW": "Gilbert",
        #         "NWAmes": "Gilbert",
        #         "ClearCr": "Crawfor",
        #         "CollgCr": "Crawfor",
        #         "Timber": "Veenker",
        #         "Somerst": "Veenker",
        #         "Edwards": "OldTown",
        #         "BrkSide": "OldTown",
        #         "StoneBr": "NridgHt",
        #         "NoRidge": "NridgHt",
        #     },
        #     inplace=True,
        # )
        #
        # self.df.SaleCondition.replace(
        #     {"AdjLand": "Abnorml", "Alloca": "Abnorml", "Family": "Abnorml"},
        #     inplace=True,
        # )
        # self.df.SaleType.replace(
        #     {
        #         "ConLD": "COD",
        #         "ConLI": "COD",
        #         "CwD": "COD",
        #         "ConLw": "COD",
        #         "Con": "COD",
        #         "Oth": "COD",
        #     },
        #     inplace=True,
        # )
        categorical_to_drop = [
            "ExterCond",
            "Fence",
            "LotConfig",
            "RoofStyle",
            "Exterior1st",
        ]
        # add columns to drop
        drop_columns = ["ExterCond", "Fence", "LotConfig", "RoofStyle", "Exterior1st"]

        # drop the selected features
        self.df.drop(columns=drop_columns, inplace=True)

        # remove the dropped columns from categorical feature list
        for cat in drop_columns[:]:
            self.categorical_features.remove(cat)

        ######### Timeseries ########
        # change the types to integer
        self.df.YrSold = self.df.YrSold.astype(int)
        self.df.GarageYrBlt = self.df.GarageYrBlt.astype(int)

        # create a derived column date sold by combining month sold and year sold
        self.df["dateSold"] = (
            self.df["MoSold"].astype(str) + "-1-" + self.df["YrSold"].astype(str)
        )
        self.df["dateSold"] = pd.to_datetime(self.df["dateSold"])
        # add the new column to the timeseries list
        self.timeseries_features.append("dateSold")

        # drop columns Mosold / Yrsold
        self.df.drop(["MoSold", "YrSold"], axis=1, inplace=True)
        for col in ["MoSold", "YrSold"]:
            self.timeseries_features.remove(col)

        # reset these categorical numerical variables to integer
        self.df[
            [
                "HalfBath",
                "Fireplaces",
                "FullBath",
                "BsmtFullBath",
                "GarageCars",
                "BedroomAbvGr",
                "OverallCond",
                "OverallQual",
            ]
        ] = self.df[
            [
                "HalfBath",
                "Fireplaces",
                "FullBath",
                "BsmtFullBath",
                "GarageCars",
                "BedroomAbvGr",
                "OverallCond",
                "OverallQual",
            ]
        ].astype(
            int
        )

        # assign the categorical columns that are non integer to categorical_columns as a list
        self.categorical_columns = [
            "ExterQual",
            "BsmtQual",
            "BsmtCond",
            "HeatingQC",
            "KitchenQual",
            "FireplaceQu",
            "GarageQual",
            "HouseStyle",
            "BsmtFinType2",
            "BsmtFinType1",
            "GarageFinish",
        ]

        # assign the labels in the order of decreasing to increase as when creating a categorical feature
        # Converting normal Object features to Categorical Data Type features
        self.df["ExterQual"] = pd.Categorical(
            self.df["ExterQual"], ordered=True, categories=["Fa", "TA", "Gd", "Ex"]
        )
        self.df["BsmtQual"] = pd.Categorical(
            self.df["BsmtQual"], ordered=True, categories=["NA", "Fa", "TA", "Gd", "Ex"]
        )
        self.df["BsmtCond"] = pd.Categorical(
            self.df["BsmtCond"], ordered=True, categories=["NA", "Po", "Fa", "TA", "Gd"]
        )
        self.df["HeatingQC"] = pd.Categorical(
            self.df["HeatingQC"], ordered=True, categories=["Po", "Fa", "TA", "Gd", "Ex"]
        )
        self.df["KitchenQual"] = pd.Categorical(
            self.df["KitchenQual"], ordered=True, categories=["Fa", "TA", "Gd", "Ex"]
        )
        self.df["FireplaceQu"] = pd.Categorical(
            self.df["FireplaceQu"],
            ordered=True,
            categories=["NA", "Po", "Fa", "TA", "Gd", "Ex"],
        )
        self.df["GarageQual"] = pd.Categorical(
            self.df["GarageQual"],
            ordered=True,
            categories=["NA", "Po", "Fa", "TA", "Gd", "Ex"],
        )
        self.df['GarageFinish'] = pd.Categorical(
            self.df['GarageFinish'], ordered=True, categories=['NA', 'Unf', 'RFn', 'Fin']
        )
        self.df['BsmtFinType1'] = pd.Categorical(
            self.df['BsmtFinType1'], ordered=True, categories=['NA', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ']
        )
        self.df['BsmtFinType2'] = pd.Categorical(
            self.df['BsmtFinType2'], ordered=True, categories=['NA', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ']
        )
        self.df['HouseStyle'] = pd.Categorical(
            self.df['HouseStyle'],
            ordered=True,
            categories=['SFoyer', '1.5Unf', '1Story', '1.5Fin', 'SLvl', '2.5Unf', '2Story', '2.5Fin']
        )

        # factorize the categories to Integer representation
        for col in self.categorical_columns:
            code, _ = pd.factorize(self.df[col],sort=True)
            self.df[col] = pd.Series(code)

        # reassign the categorical features
        self.categorical_features = list(self.df.select_dtypes(include=[np.object]).columns.values)

        # created dummy variables for categorical features
        self.house_price = pd.concat(
            [self.df, pd.get_dummies(self.df[self.categorical_features], drop_first=True)], axis=1
        )

        # drop the actual categorical feature from list
        self.house_price.drop(columns=self.categorical_features, inplace=True)
        # reset index for the new dataframe
        self.house_price.reset_index(drop=True, inplace=True)

        ######## TIME ########
        # lets create a constant time
        self.tm = datetime.time(10, 10)

        # convert the dateSold to unix timestamp
        self.house_price.dateSold = self.house_price.dateSold.apply(
            lambda x: datetime.datetime.combine(x, self.tm).timestamp()
        )

        # reassigning all the numerical features to the numerical_features variable as a list
        self.numerical_features = list(self.df.select_dtypes(include=[np.number]).columns.values)
        print(self.house_price.shape)
        return self.df

    def get_raw_df(self):
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
        self.df["TotalSF"] = self.df["TotalBsmtSF"] + self.df["1stFlrSF"] + self.df["2ndFlrSF"]

        self.df["LivLotRatio"] = self.df.GrLivArea / self.df.LotArea

        self.df["Spaciousness"] = (self.df.FirstFlrSF + self.df.SecondFlrSF) / self.df.TotRmsAbvGrd

        # Adding all the bathrooms
        self.df["TotalBath"] = self.df[
            ["BsmtFullBath", "BsmtHalfBath", "FullBath", "HalfBath"]
        ].sum(axis=1)

        # Adding square feet of all Porch
        self.df["TotalPorchSF"] = self.df[
            ["OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch", "WoodDeckSF"]
        ].sum(axis=1)

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
