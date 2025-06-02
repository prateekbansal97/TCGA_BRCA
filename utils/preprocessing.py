import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_rnaseq(file_path):
    df = pd.read_csv(file_path, sep='\t', index_col=0)
    df = df.T                                                                           # Samples as rows
    df.index.name = 'sample'                                                            #To be able to find common samples
    return df

def load_clinical_phenotype(file_path):
    df = pd.read_csv(file_path, sep='\t', index_col=0)
    df = df[df['vital_status'].notna() & df['days_to_last_followup'].notna()]           #Remove nans
    df['event'] = df['vital_status'].apply(lambda x: 1 if x == 'DECEASED' else 0)
    df['time'] = df['days_to_last_followup']
    df.index.name = 'sample'                                                            #To be able to find common samples
    return df[['event', 'time']]


def generate_feature_and_labels(rna_df, clinical_phenotype_df):
    joined = rna_df.join(clinical_phenotype_df, how='inner', on='sample')               #intersection of two dfs
    y = joined[['event', 'time']].copy()
    X = joined.drop(columns=['event', 'time'])
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)  #Normalizing
    return X_scaled, y
