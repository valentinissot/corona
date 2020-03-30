corona
==============================

Projet Coronavirus

Organisation du projet
------------

Notebooks: Ensemble des notebooks jupyter pour l'étude et le traitement des données du coronavirus.

    - Corona_collecte_donnees.ipynb: Récupération des données et mis en forme d'une table réunissant l'ensemble des données.
    
    - Corona_visualisation.ipynb: Ensemble de graphiques permettant l'étude comparative de l'évolution des données par pays.
    
    - Corona_modélisation.ipynb: Application du modèle SIR sur les données. 
    
Src - app.py: Réalisation d'une application Dash avec trois pages pour la visualisation graphique de l'évolution des données 
et du modèle SIR utilisé dans la partie "Notebooks".



    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    │
    ├── \_can_be_deleted   <- Trash bin (!! git ignored)
    │
    ├── confidential       <- Confidential documents, data, etc. (!! git ignored)
    │
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   ├── raw            <- The original, immutable data dump.
    │   └── working        <- Working, intermediate data that has been transformed.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │                         Also includes sklearn & pyspark pipelines.
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-rvm-initial-data-exploration`.
    │
    ├── production
    │   ├── config         <- YAML files with dependancies between tasks, data catalog and others.
    │   ├── pipelines      <- sklearn & pyspark pipelines.
    │   ├── tasks          <- Luigi tasks.
    │   └── scripts        <- Functions used by Luigi tasks.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org


--------

<p><small>Project based on the <a target="_blank" href="http://git.equancy.io/tools/cookiecutter-data-science-project/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
