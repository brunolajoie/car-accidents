import pandas as pd
import numpy as np

def build_dataset(datapath):

    # Load files
    cara = pd.read_csv(f"{datapath}caracteristics.csv")
    users = pd.read_csv(f"{datapath}users.csv")
    places = pd.read_csv(f"{datapath}places.csv")
    vehicles = pd.read_csv(f"{datapath}vehicles.csv")

    # Merge caracteristics and places on 'Num_Acc'
    data = cara.merge(places, on='Num_Acc')

    # Create a common key to merge users and vehicles on
    users['Num_Acc_num_veh'] = users['Num_Acc'].map(lambda x: str(x)) + users['num_veh']
    vehicles['Num_Acc_num_veh'] = vehicles['Num_Acc'].map(lambda x: str(x)) + vehicles['num_veh']
    
    # Remove useless columns
    vehicles = vehicles.drop(columns=['index'])
    users = users.drop(columns=['index', 'Num_Acc', 'num_veh'])
    # Merge vehicles and users
    tmp = vehicles.merge(users, on='Num_Acc_num_veh', how='inner')
    
    data = data.merge(tmp, on='Num_Acc', how='inner')
    
    print(f"Dataset has {len(data)} rows.")
    
    return data

def clean_dataset(data_, useless_features):
    # Drop lines without targets (if any)
    df = data_[~np.isnan(data_['grav'])]
    df = df.drop(columns=useless_features)
    # Drop lines without 'secu'
    df = df[~np.isnan(df['secu'])]
    # Only keep rows with "secu" number consisting of two digits
    df = df[df['secu'].map(lambda x: len(str(round(x)))) == 2]
    return df
