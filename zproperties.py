import torch

class ZProperties:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocabThreshold = 1
    maxLength = 64
    embedSize = 128
    numHeads = 4
    numLayers = 2
    dimFeedforward = 512
    dropout = 0.2

    tabularEmbedSize = 64
    fusionHiddenSize = 256

    batchSize = 64
    learningRate = 2e-4
    weightDecay = 1e-2
    numEpochs = 30

    trainSplit = 0.8
    seed = 42
    modelPath = "model.pth"
    datasetPath = "dataset.pkl"
