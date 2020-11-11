import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from load import build_dataset, clean_dataset
from pipeline import build_pipeline
from utils import binarize_target

DATAPATH = 'data/data_training/'

USELESS_FEATURES = [
    'locp', 'actp', 'etatp', 'v2', 'lat', 'long', 'gps',
    'pr1', 'pr', 'v1', 'adr', 'voie', 'index_x', 'Num_Acc',
    'Num_Acc_num_veh', 'Num_Acc', 'num_veh', 'index_y',
    'jour', 'an', 'dep', 'com', 'env1'
]

NUMERICAL_FEATURES = ['nbv', 'senc', 'an_nais', 'occutc', 'lartpc', 'larrout']

CATEGORICAL_FEATURES = [
    'hour_of_day', 'is_safety_equipment', 'safety_equipment', 'surf', 'prof',
    'place', 'manv', 'circ', 'lum', 'catv', 'obsm', 'infra', 'agg', 'atm', 'catr',
    'situ', 'obs', 'vosp', 'catu', 'int', 'trajet', 'sexe', 'plan', 'choc', 'col'
]

CYCLICAL_FEATURES =  ['hour_of_day', 'mois']

##################### WORKFLOW #####################

# Load data
data = build_dataset(DATAPATH)
data_cleaned = clean_dataset(data, USELESS_FEATURES)

# Hold-out
X, y = data_cleaned.drop(columns=['grav']), binarize_target(data_cleaned['grav'])
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3)

# Preprocessing and fitting pipeline
pipe = build_pipeline(
    numerical_features=NUMERICAL_FEATURES,
    categorical_features=CATEGORICAL_FEATURES,
    cyclical_features=CYCLICAL_FEATURES
)

pipe.fit(X_train, y_train)

# Evaluation
print(classification_report(pipe.predict(X_test), y_test))
