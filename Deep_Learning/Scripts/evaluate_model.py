import torch
from src.evaluate import Evaluator
from src.model import StructuredNN

def main():

    model = StructuredNN(input_dim=20)
    model.load_state_dict(torch.load("../models/model.pt"))

    evaluator = Evaluator(model)

    evaluator.evaluate()

if __name__ == "__main__":
    main()