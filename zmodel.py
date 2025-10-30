import torch
import torch.nn as nn
from zproperties import ZProperties

class PositionalEncoding(nn.Module):
    def __init__(self, embedSize, maxLen):
        super().__init__()
        pe = torch.zeros(maxLen, embedSize)
        position = torch.arange(0, maxLen, dtype=torch.float).unsqueeze(1)
        divTerm = torch.exp(torch.arange(0, embedSize, 2).float() * (-torch.log(torch.tensor(10000.0)) / embedSize))
        pe[:, 0::2] = torch.sin(position * divTerm)
        pe[:, 1::2] = torch.cos(position * divTerm)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerTextEncoder(nn.Module):
    def __init__(self, vocabSize, embedSize, numHeads, numLayers, dimFeedforward, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocabSize, embedSize)
        self.posEncoding = PositionalEncoding(embedSize, ZProperties.maxLength)
        encoderLayer = nn.TransformerEncoderLayer(embedSize, numHeads, dimFeedforward, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoderLayer, numLayers)
        self.layerNorm = nn.LayerNorm(embedSize)

    def forward(self, x):
        x = self.embedding(x)
        x = self.posEncoding(x)
        x = self.encoder(x)
        x = self.layerNorm(x.mean(dim=1))
        return x

class InteractionModule(nn.Module):
    def __init__(self, embedSize):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(embedSize * 2, embedSize),
            nn.GELU(),
            nn.Linear(embedSize, embedSize),
            nn.LayerNorm(embedSize)
        )

    def forward(self, a, b):
        combined = torch.cat([a, b], dim=1)
        return self.fc(combined)

class TabularBranch(nn.Module):
    def __init__(self, inputSize, embedSize):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(inputSize, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Dropout(0.2),
            nn.Linear(128, embedSize),
            nn.GELU()
        )

    def forward(self, x):
        return self.fc(x)

class FusionClassifier(nn.Module):
    def __init__(self, textDim, tabDim, hiddenDim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(textDim + tabDim, hiddenDim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hiddenDim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, text, tab):
        x = torch.cat([text, tab], dim=1)
        return self.fc(x)

class SalesPredictor(nn.Module):
    def __init__(self, vocabSize, tabInputSize):
        super().__init__()
        self.textEncoder = TransformerTextEncoder(
            vocabSize, ZProperties.embedSize,
            ZProperties.numHeads, ZProperties.numLayers,
            ZProperties.dimFeedforward, ZProperties.dropout
        )
        self.interaction = InteractionModule(ZProperties.embedSize)
        self.tabularBranch = TabularBranch(tabInputSize, ZProperties.tabularEmbedSize)
        self.classifier = FusionClassifier(
            ZProperties.embedSize, ZProperties.tabularEmbedSize, ZProperties.fusionHiddenSize
        )

    def forward(self, xTextA, xTextB, xTab):
        a = self.textEncoder(xTextA)
        b = self.textEncoder(xTextB)
        textFeatures = self.interaction(a, b)
        tabFeatures = self.tabularBranch(xTab)
        return self.classifier(textFeatures, tabFeatures)
