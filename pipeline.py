import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, StandardScaler
from transformers import DataFrameConverter, HourParser, SafetyEncoder
from sklearn.ensemble import RandomForestClassifier

def build_pipeline_numerical(numerical_features):

    # List steps that we want to apply to our numerical features.
    steps = [
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ]

    # Use ColumnTransformer to specify the columns that will be preprocessed. 
    preprocessor = ColumnTransformer([
        ('imputer_scaler', Pipeline(steps), numerical_features)
    ])

    # We could just return `preprocessor` but if we want to keep the column names, we can!
    return Pipeline([
        ('preprocessing', preprocessor),
        ('conversion', DataFrameConverter(column_names=numerical_features))
    ])    

def build_pipeline_categorical(categorical_features):

    # List steps that we want to apply to our categorical features.
    # OneHotEncoding does not work with missing values, imputation is required.
    steps = [
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder())
    ]

    # Use ColumnTransformer to specify the columns that will be preprocessed.
    preprocessor = ColumnTransformer([
        ('imputer_encoder', Pipeline(steps), categorical_features)
    ])

    # Same here, we could just return `preprocessor` but let us say we want to 
    # work with a DataFrame.
    return Pipeline([
        ('preprocessing', preprocessor),
        ('conversion', DataFrameConverter())
    ])

def build_pipeline_cyclical(cyclical_features):

    # Remember that ColumnTransformer applies each processing step in parallel.
    preprocessor = ColumnTransformer([
        ('sin_transform', FunctionTransformer(np.sin), cyclical_features),
        ('cos_transform', FunctionTransformer(np.cos), cyclical_features),
    ])
    
    # You MUST know how ColumnTransformer transforms your input so you can
    # rename your columns.
    column_names = (
        [feature_name + '_sin' for feature_name in cyclical_features] + 
        [feature_name + '_cos' for feature_name in cyclical_features]
    )

    return Pipeline([
        ('preprocessing', preprocessor),
        ('conversion', DataFrameConverter(column_names=column_names))
    ])

def build_pipeline(numerical_features, categorical_features, cyclical_features):
    '''
    Pipeline steps:
    - Create a new feature called `hour_of_day`
    - Create two new features called `safety_equipment` and `is_safety_equipment`
    - Impute and scale numerical features
    - Impute and one-hot encode categorical features
    - Apply sin and cos transformations to `hour_of_day` and `mois`
    - Fit a Random Forest model
    '''
    pipes = {
        "numerical": build_pipeline_numerical(numerical_features=numerical_features),
        "categorical": build_pipeline_categorical(categorical_features=categorical_features),
        "cyclical": build_pipeline_cyclical(cyclical_features=cyclical_features)
    }
    
    preprocessor = FeatureUnion([
        ("preprocessing_num", pipes["numerical"]),
        ("preprocessing_cat", pipes["categorical"]),
        ("preprocessing_cyc", pipes["cyclical"]),
    ])

    return Pipeline([
        ('hour_parser', HourParser()),
        ('safety_encoder', SafetyEncoder()),
        ('preprocessing', preprocessor),
        ('rf_classifier', RandomForestClassifier(class_weight='balanced'))
    ])
