import pandas as pd
import pickle
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from zproperties import ZProperties
import torch
from torch.utils.data import Dataset

def buildVocab(texts, threshold=1):
    counter = Counter()
    for t in texts:
        counter.update(t.lower().split())

    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, freq in counter.items():
        if freq >= threshold:
            vocab[word] = len(vocab)
    return vocab

def textToSeq(text, vocab, maxLen):
    tokens = text.lower().split()
    seq = [vocab.get(tok, vocab["<UNK>"]) for tok in tokens][:maxLen]
    if len(seq) < maxLen:
        seq += [vocab["<PAD>"]] * (maxLen - len(seq))
    return seq

class OpportunityDataset(Dataset):
    def __init__(self, data, vocab, scaler, labelEncoder):
        self.data = data
        self.vocab = vocab
        self.scaler = scaler
        self.labelEncoder = labelEncoder

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        xTextA = torch.tensor(textToSeq(row["conversa_inicial"], self.vocab, ZProperties.maxLength))
        xTextB = torch.tensor(textToSeq(row["conversa_final"], self.vocab, ZProperties.maxLength))
        tabular = torch.tensor(row["tabular"], dtype=torch.float32)
        y = torch.tensor(row["label"], dtype=torch.float32)
        return xTextA, xTextB, tabular, y

def createDataset(csvPath):
    df = pd.read_csv(csvPath)

    vocab = buildVocab(df["conversa_inicial"].tolist() + df["conversa_final"].tolist(), threshold=ZProperties.vocabThreshold)

    labelEncoder = LabelEncoder()
    catCols = ["produto", "setor_cliente", "tamanho_cliente"]
    for c in catCols:
        df[c] = labelEncoder.fit_transform(df[c])

    numCols = ["valor_proposta", "tempo_negociacao", "num_reunioes"]
    scaler = StandardScaler()
    df[numCols] = scaler.fit_transform(df[numCols])

    df["tabular"] = df[numCols + catCols].values.tolist()
    df["label"] = df["status_final"].apply(lambda x: 1 if str(x).strip().lower() == "venda" else 0)

    trainDf, valDf = train_test_split(df, test_size=1 - ZProperties.trainSplit, random_state=ZProperties.seed)

    dataset = {
        "train": trainDf.reset_index(drop=True),
        "val": valDf.reset_index(drop=True),
        "vocab": vocab,
        "scaler": scaler,
        "labelEncoder": labelEncoder
    }

    with open(ZProperties.datasetPath, "wb") as f:
        pickle.dump(dataset, f)

    print(f"Dataset saved to {ZProperties.datasetPath} with {len(vocab)} tokens.")

    return dataset
