import torch
from torch.utils.data import DataLoader
from torch import nn, optim
import pickle
from model import SalesPredictor
from zdataset import OpportunityDataset
from zproperties import ZProperties
from tqdm import tqdm

def trainModel():
    with open(ZProperties.datasetPath, "rb") as f:
        data = pickle.load(f)

    trainSet = OpportunityDataset(data["train"], data["vocab"], data["scaler"], data["labelEncoder"])
    valSet = OpportunityDataset(data["val"], data["vocab"], data["scaler"], data["labelEncoder"])

    trainLoader = DataLoader(trainSet, batch_size=ZProperties.batchSize, shuffle=True)
    valLoader = DataLoader(valSet, batch_size=ZProperties.batchSize)

    model = SalesPredictor(len(data["vocab"]), len(data["train"]["tabular"][0])).to(ZProperties.device)
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=ZProperties.learningRate, weight_decay=ZProperties.weightDecay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=ZProperties.numEpochs)

    for epoch in range(ZProperties.numEpochs):
        model.train()
        totalLoss = 0
        for xA, xB, tab, y in tqdm(trainLoader, desc=f"Epoch {epoch+1}/{ZProperties.numEpochs}"):
            xA, xB, tab, y = xA.to(ZProperties.device), xB.to(ZProperties.device), tab.to(ZProperties.device), y.to(ZProperties.device).unsqueeze(1)
            preds = model(xA, xB, tab)
            loss = criterion(preds, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            totalLoss += loss.item()
        scheduler.step()

        model.eval()
        valLoss = 0
        with torch.no_grad():
            for xA, xB, tab, y in valLoader:
                xA, xB, tab, y = xA.to(ZProperties.device), xB.to(ZProperties.device), tab.to(ZProperties.device), y.to(ZProperties.device).unsqueeze(1)
                preds = model(xA, xB, tab)
                valLoss += criterion(preds, y).item()

        print(f"Epoch {epoch+1}: Train Loss = {totalLoss/len(trainLoader):.4f} | Val Loss = {valLoss/len(valLoader):.4f}")

    torch.save(model.state_dict(), ZProperties.modelPath)
    print(f"Model saved to {ZProperties.modelPath}")

if __name__ == "__main__":
    trainModel()
