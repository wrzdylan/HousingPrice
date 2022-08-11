# Exercice RNCP régression linéaire. 


## Contexte
Vous travaillez pour la société jachetetoutlimmobilier. Cette dernière veut repérer les maisons intéressantes à acheter selon un algorithme de machine learning. 

Pour cela elle a fait travailler un datascientist qui a préparé le set qui contient 82 variables. 


## Objectifs
* Refactorisez la classe de préparation du set, en comprenant bien les étapes séquentielles de préparation de la data et en pouvant les expliquer. 
* Sélectionnez les features ayant le plus d'importance pour votre précision et expliquez votre méthode de sélection.
* Prédisez les prix de l'ensemble de votre test set. Prévoyez au minimum le test de deux algorithmes ainsi que le fine tuning des paramètres de ces deux algorithmes. 


## Methodologie
> [Lien Kaggle](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data?select=data_description.txt)

### Sélection des features
- [EDA](https://www.kaggle.com/code/pmarcelino/comprehensive-data-exploration-with-python)
- https://www.kaggle.com/code/serigne/stacked-regressions-top-4-on-leaderboard
Etudier le SalePrice.      
Pour la sélection des features, voir **ACP**.     
Se demander si on pense à cette feature quand on achète une maison et quelle est son importance ? A ÉVITER car trop subjectif.     
Faire la corrélation entre les features et SalePrice via un scatterplot par exemple et un boxplot pour les catégories. Heatmap ?        

Missing data, random ou pattern ? 15% de données manquants sont trop, autant supprimer ces valeurs.    
Faire attention aux outliers et les gérer car donne des informations mais peut modifier le comportement d'un model.


### Prediction
Approche linéaire :
- https://www.kaggle.com/code/apapiu/regularized-linear-models


## Clean
- Ajoute log SalePrice in cleaned data
- Missing values
  - [X] Remplace NA par None        
  - [X] Sauf LotFrontage, on fait la médiane de cette variable par rapport au voisinage car probablement similaire
  - [X] Pour 'MasVnrArea', 'GarageYrBlt', 'GarageArea', 'GarageCars', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath' remplace NA par 0         
  - [X] Functional by Typ          
  - [X] Peut drop 'Utilities' car uniquement la même valeur           
  - [X] MSZoning, Electrical, SaleType, KitchenQual, Exterior1st and Exterior2nd  donne la valeur la plus fréquente avec `all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])`          
- Transform variables types
  - Numérique into categorical : MSSubClass, OverallCond, YrSold, MoSold -> `all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)`
  - LabelEncoder some categorical variables :
      ```python
      from sklearn.preprocessing import LabelEncoder
      cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
              'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
              'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
              'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
              'YrSold', 'MoSold')
      # process columns, apply LabelEncoder to categorical features
      for c in cols:
          lbl = LabelEncoder() 
          lbl.fit(list(all_data[c].values)) 
          all_data[c] = lbl.transform(list(all_data[c].values))
    
      # shape        
      print('Shape all_data: {}'.format(all_data.shape))
      ```
- Ajoute feature `all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']` car influe beaucoup sur le prix
- Regarde le skew de chaque feature numeric et la rend plus normale :
```python
numeric_feats = df.dtypes[df.dtypes != "object"].index

# Check the skew of all numerical features
skewed_feats = df[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
skewness = pd.DataFrame({'Skew' :skewed_feats})

skewed_features = skewness.index
df[skewed_features] = np.log1p(df[skewed_features])
```
- getting dummy categorical features, valeur numérique qui représente une catégorie
```python
df = pd.get_dummies(df)
print(df.shape)
```


## Lexique
- dummy variables, valeurs numériques qui représentent des données de catégories
- [diff between apply and transform](https://towardsdatascience.com/difference-between-apply-and-transform-in-pandas-242e5cf32705)
  - transform() peut utiliser des function strings, des list de functions (ex: `df.transform([np.sqrt, np.exp])`) et des dictionnaires de functions
  - transform() ne peut pas produire des résultats aggrégés (ex: `df.transform(lambda x:x.sum())`)
  - transform() ne peut pas travailler avec +sieurs Series en même temps (ex: `df.transform(subtract_two, axis=1)` où subtract_two fait une vectorization entre 2 colonnes)
  - Avec un **groupby()**, transform() renvoie une série avec la même longueur, c'est son **principale avantage**.
-  La fonction pandas `.mode()` retourne la valeur la plus commune de chaque colonnes