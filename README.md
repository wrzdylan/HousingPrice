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
- [Explication des algos](https://www.kaggle.com/code/faressayah/practical-introduction-to-10-regression-algorithm)
- Voir [link](bayesian optimization package)


### Sélection des features
- [EDA](https://www.kaggle.com/code/pmarcelino/comprehensive-data-exploration-with-python)
- [Sélection des features](https://www.kaggle.com/code/prashant111/comprehensive-guide-on-feature-selection)
- [idem](https://machinelearningmastery.com/feature-selection-with-real-and-categorical-data/)
- https://www.kaggle.com/code/serigne/stacked-regressions-top-4-on-leaderboard
Etudier le SalePrice.      
Pour la sélection des features, voir **ACP**.     
Se demander si on pense à cette feature quand on achète une maison et quelle est son importance ? A ÉVITER car trop subjectif.     

Missing data, random ou pattern ? 15% de données manquants sont trop, autant supprimer ces valeurs.    
Faire attention aux outliers et les gérer car donne des informations mais peut modifier le comportement d'un model.

La sélection des features permet d'améliorer la précision du model, réduit l'overfitting, plus facile et rapide pour le model 


#### Filter methods
Utilise des stats et regarde l'importance de chaque features par rapport à la target

##### Basic methods
On enlève les features avec quasiment les mêmes valeurs (peut utiliser VarianceThreshold de sklearn)

##### Univariatge selection methods
Utilise des tests statistiques univariate, estime le degré de dépendance linéaire entre 2 variables random, doit suivre une distribution Gaussienne
Si numérique en input et output : Pearson’s correlation pour linear et Spearman’s rank coefficient dans le cas contraire

Pearson’s correlation mesure la force d'une relation linéaire entre 2 variables, a une valeur entre -1 et 1
Ratio entre la covariance de deux variables et le produit de leurs écart-type. C'est une mesure normalisé de la covariance.

Les méthodes les plus populaire sont :
- Select the top k variables: SelectKBest
- Select the top percentile variables: SelectPercentile

#### Wrapper methods
RFE
Try to use a subset of features and train a model using them. 
Based on the inferences that we draw from the previous model, we decide to add or remove features from the subset.

#### Embedded methods
Takes care of each iteration of the model training process and carefully extract those features which contribute the most to the training for a particular iteration.
Regularization methods are the most commonly used embedded methods which penalize a feature given a coefficient threshold.

Most popular methods are: LASSO and RIDGE regression


### Prediction
Approche linéaire :
- https://www.kaggle.com/code/apapiu/regularized-linear-models
XGBoost :
- https://www.kaggle.com/code/ryanholbrook/feature-engineering-for-house-prices
Neural network:
- https://www.kaggle.com/code/zoupet/neural-network-model-for-house-prices-tensorflow


## Clean
- [X] Ajoute log SalePrice in cleaned data
- [X] Missing values
  - [X] Remplace NA par None        
  - [X] Sauf LotFrontage, on fait la médiane de cette variable par rapport au voisinage car probablement similaire
  - [X] Pour 'MasVnrArea', 'GarageYrBlt', 'GarageArea', 'GarageCars', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath' remplace NA par 0         
  - [X] Functional by Typ          
  - [X] Peut drop 'Utilities' car uniquement la même valeur           
  - [X] MSZoning, Electrical, SaleType, KitchenQual, Exterior1st and Exterior2nd  donne la valeur la plus fréquente avec .mode()          
- [X] Transform variables types
  - [X] Numérique into categorical : MSSubClass, OverallCond, YrSold, MoSold -> `all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)`
  - [X] Ordinal Encoder pour certaine variable:
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
- [X] Ajoute feature 'totalSF' car influe beaucoup sur le prix
- [X] Regarde le skew de chaque feature numeric et la rend plus normale 
- [X] Dummys
- [] Ajouter quelques features liées aux date, par exemple, différence entre date de construction, date d'agrandissement et maintenant
- [] Prendre en compte le test set pour tout ce qui est encoding, ...
- [] Drop les features avec une corrélation trop forte entre elles (ne pas prendre en compte SalePrice) car peut créer de l'overfitting


log1p vs boxcox1p lequel utiliser ?
normalize avant ou après Ordinal Encoder ?

## Amélioration possible
Prendre en compte l'inflation et le pouvoir d'achat


## Lexique
- dummy variables, valeurs numériques qui représentent des données de catégories
- [diff between apply and transform](https://towardsdatascience.com/difference-between-apply-and-transform-in-pandas-242e5cf32705)
  - transform() peut utiliser des function strings, des list de functions (ex: `df.transform([np.sqrt, np.exp])`) et des dictionnaires de functions
  - transform() ne peut pas produire des résultats aggrégés (ex: `df.transform(lambda x:x.sum())`)
  - transform() ne peut pas travailler avec +sieurs Series en même temps (ex: `df.transform(subtract_two, axis=1)` où subtract_two fait une vectorization entre 2 colonnes)
  - Avec un **groupby()**, transform() renvoie une série avec la même longueur, c'est son **principale avantage**.
- La fonction pandas `.mode()` retourne la valeur la plus commune de chaque colonnes
- A nominal variable is the same as categorical, has two or more variables but there is no instrinsic ordering to the categories (ex: binary questions)
- An ordinal variable has a clear ordering of the categories (ex: wages)
- `.factorize()` transforme des valeurs nominales en numériques mais continues, un model va alors donner plus ou moins d'importance en fonction de sa grandeur
Pour cette raison, on utilise plutôt le `One-Hot Encoding`, celui-ci va créer plusieurs features et les remplir avec des 0 et des 1, il y a néanmoins une forte augmentation des dimensions
- Le Label Encoder permet de convertir une variable nominale en variable numérique ordonnée. 
  - OrdinalEncoder is for 2D data with the shape (n_samples, n_features)
  - LabelEncoder is for 1D data with the shape (n_samples,)
- Le One-Hot Encoding crée une variable binaire pour chaque catégorie mais cette représentation est redondante car on a pas besoin d'une variable pour chaque catégorie pour savoir dans laquelle elle appartient
- Dummy variable représente C catégories avec C-1 variables binaires, doit être utilisé pour certain model comme les régressions linéaires