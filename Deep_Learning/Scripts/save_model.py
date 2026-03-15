import torch

def save_model(model):

    torch.save(
        model.state_dict(),
        "../models/hotel_cancellation_model.pt"
    )