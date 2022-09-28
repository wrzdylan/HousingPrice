# Cas pratique - Housing Price


## Contexte
Vous travaillez pour la société jachetetoutlimmobilier. Cette dernière veut repérer les maisons intéressantes à acheter selon un algorithme de machine learning. 

Pour cela elle a fait travailler un datascientist qui a préparé le set qui contient 82 variables. 


## Objectifs
* Refactorisez la classe de préparation du set, en comprenant bien les étapes séquentielles de préparation de la data et en pouvant les expliquer. 
* Sélectionnez les features ayant le plus d'importance pour votre précision et expliquez votre méthode de sélection.
* Prédisez les prix de l'ensemble de votre test set. Prévoyez au minimum le test de deux algorithmes ainsi que le fine tuning des paramètres de ces deux algorithmes. 


## Architecture
- Le premier notebook, `data_processing.ipynb` contient l'analyse de données, on y regarde la distribution des variables, leurs relations, le nombre de valeurs manquantes ou encore les outliers.     
De cette première étape, dépend le pipeline de données et la sélection des algorithmes.        
- Le script `main.py` lance le traitement des données pour le training et test sets via le module GetDataFrame.       
- Le module `GetDataFrame.py`, on remplace les valeurs manquantes, on ajoute de nouvelles features, modifie les outliers puis transforme les variables.     
- Enfin le notebook `training_models.ipynb`, entraine les algorithmes LinearRegression, Lasso, Ridge, ElasticNet et XGBoost, les indicateurs de performances sont stockés dans le fichier `regression_models.csv`.      
