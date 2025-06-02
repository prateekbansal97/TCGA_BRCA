import pandas as pd

def load_rnaseq(file_path):
    df = pd.read_csv(file_path, sep='\t', index_col=0)
    df = df.T  # Samples as rows
    df.index.name = 'sample'
    return df

def load_clinical_phenotype(file_path):
    df = pd.read_csv(file_path, sep='\t', index_col=0)
    df = df[df['vital_status'].notna() & df['days_to_last_followup'].notna()]
    df['event'] = df['vital_status'].apply(lambda x: 1 if x == 'DECEASED' else 0)
    df['time'] = df['days_to_last_followup']
    df.index.name = 'sample'
    return df[['event', 'time']]
