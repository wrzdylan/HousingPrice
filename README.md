# Exercice RNCP régression linéaire. 

## Contexte

Vous travaillez pour la société jachetetoutlimmobilier. Cette dernière veut repérer les maisons intéressantes à acheter selon un algorithme de machine learning. 

Pour cela elle a fait travailler un datascientist qui a préparé le set qui contient 82 variables. 

## Objectifs 

* Refactorisez la classe de préparation du set, en comprenant bien les étapes séquentielles de préparation de la data et en pouvant les expliquer. 
* Sélectionnez les features ayant le plus d'importance pour votre précision et expliquez votre méthode de sélection. => ACP
* Prédisez les prix de l'ensemble de votre test set. Prévoyez au minimum le test de deux algorithmes ainsi que le fine tuning des paramètres de ces deux algorithmes. 

## Methodologie
> [Lien Kaggle](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data?select=data_description.txt)

### Sélection des features
- [EDA](https://www.kaggle.com/code/pmarcelino/comprehensive-data-exploration-with-python)
Etudier le SalePrice.      
Pour la sélection des features, voir **ACP**.     
Se demander si on pense à cette feature quand on achète une maison et quelle est son importance ? A ÉVITER car trop subjectif.     
Faire la corrélation entre les features et SalePrice via un scatterplot par exemple et un boxplot pour les catégories. Heatmap ?        

Missing data, random ou pattern ? 15% de données manquants sont trop, autant supprimer ces valeurs.    
Faire attention aux outliers et les gérer car donne des informations mais peut modifier le comportement d'un model.

### Prediction
Approche linéaire :
- https://www.kaggle.com/code/apapiu/regularized-linear-models

