import pandas as pd

def load_rnaseq(file_path):
    df = pd.read_csv(file_path, sep='\t', index_col=0)
    df = df.T  # Samples as rows
    df.index.name = 'sample'
    return df

