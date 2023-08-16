# OC_DS_P7_implementez_modele_scoring
Projet n°7 - Parcours data scientist - OpenClassrooms

## Implémentez un modèle de scoring (prêt bancaire)

La société financière, nommée __"Prêt à dépenser"__, propose des crédits à la consommation pour des personnes ayant peu ou pas du tout d'historique de prêt.

L’entreprise souhaite __mettre en oeuvre un outil de scoring crédit pour calculer la probabilité de défaut de paiement du client__ pour décider d'accorder ou non un prêt à un client potentiel en s’appuyant sur des sources de données variées (données comportementales, données provenant d'autres institutions financières, etc.).

De plus, les chargés de relation client ont fait remonter le fait que les clients sont de plus en plus demandeurs de __transparence__ vis-à-vis des décisions d’octroi de crédit. Cette demande de transparence des clients va tout à fait dans le sens des valeurs que l’entreprise veut incarner.

Elle décide donc de développer un __dashboard interactif__ pour que les chargés de relation client puissent à la fois expliquer de façon la plus transparente possible les décisions d’octroi de crédit, mais également permettre à leurs clients de disposer de leurs informations personnelles et de les explorer facilement. 

## Données
[Source des données](https://www.kaggle.com/c/home-credit-default-risk/data)


## Mission
- Construire un modèle de scoring qui donnera une prédiction sur la probabilité de faillite d'un client de façon automatique.

- Construire un dashboard interactif à destination des gestionnaires de la relation client permettant d'interpréter les prédictions faites par le modèle et d’améliorer la connaissance client des chargés de relation client.

- Mettre en production le modèle de scoring de prédiction à l’aide d’une API, ainsi que le dashboard interactif qui appelle l’API pour les prédictions.


## Spécifications du dashboard
Il devra contient les fonctionnalités suivantes :

- Permets de visualiser le score et l’interprétation de ce score pour chaque client de façon intelligible pour une personne non experte en data science.
- Permets de visualiser des informations descriptives relatives à un client (via un système de filtre).
- Permets de comparer les informations descriptives relatives à un client à l’ensemble des clients ou à un groupe de clients similaires.

## Spécifications techniques

- Afin de pouvoir faire évoluer régulièrement le modèle, on tesetera la mise en oeuvre d’une démarche de type MLOps d’automatisation et d’industrialisation de la gestion du cycle de vie du modèle.

- Utilisation de la librairie evidently pour détecter dans le futur du Data Drift en production. 

- Le déploiement de l'application dashboard et de l’API ont été réalisées sur AWS.

- La démarche d’élaboration des modèles utilise Cross-Validation, via GridsearchCV.

- Les valeurs de ‘accuracy’ sont surveillés pour veiller sur l’overfitting.

- Une note technique acompagne l'ensemble du project, présentant la démarche d’élaboration du modèle jusqu’à l’analyse du Data Drift. 
