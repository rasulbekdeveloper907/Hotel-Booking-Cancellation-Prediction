import torch
import pandas as pd
from src.model import StructuredNN

def predict(sample):

    model = StructuredNN(input_dim=len(sample))
    model.load_state_dict(torch.load("../models/model.pt"))

    model.eval()

    x = torch.tensor(sample).float().unsqueeze(0)

    with torch.no_grad():
        prob = torch.sigmoid(model(x))

    return prob.item()