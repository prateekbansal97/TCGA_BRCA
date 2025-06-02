import torch
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
from utils.preprocessing import load_rnaseq, load_clinical_phenotype, generate_feature_and_labels
from models.mlp import MLP
from utils.metrics import evaluate
from sklearn.model_selection import train_test_split
from tqdm import tqdm

#Input data files
rna_data_path = './data/RNAseq_HiSeqV2.tsv'
clinical_data_path = './data/Clinical_Phenotype.tsv'
N_EPOCHS = 10
learning_rate = 1e-3

rna_data = load_rnaseq(rna_data_path)
clinical_data = load_clinical_phenotype(clinical_data_path)

#Naive splitting
features, labels = generate_feature_and_labels(rna_data, clinical_data)
labels = labels['event'].values                                                     #ALIVE OR DECEASED

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=np.random.randint(10000))

#print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

tensor_x_train = torch.tensor(X_train.values, dtype=torch.float32)
tensor_y_train = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)          #Torch needs this shape
tensor_x_test = torch.tensor(X_test.values, dtype=torch.float32)
tensor_y_test = torch.tensor(y_test.reshape(-1, 1), dtype=torch.float32)

train_dataset = TensorDataset(tensor_x_train, tensor_y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

model = MLP(input_dim=X_train.shape[1])                                             #20530
criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# Training loop
for epoch in tqdm(range(N_EPOCHS), leave=False):
    model.train()
    for xb, yb in train_loader:
        pred = model(xb)
        loss = criterion(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")


model.eval()
with torch.no_grad():
    pred = model(tensor_x_test).numpy().flatten()
    scores = evaluate(y_test, pred)
    print("Test Accuracy:", scores['accuracy'])
    print("Test ROC-AUC:", scores['roc_auc'])
