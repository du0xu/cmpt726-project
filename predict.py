"""
Run this script to predict outputs given new inputs, using the trained model.
"""
from argparse import ArgumentParser

import torch

from model import KeypointPredictionModel


def predict(model_path):
    # Use CUDA (GPU) if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Initialize the model and load the trained parameters
    model = KeypointPredictionModel()
    model.load_state_dict(torch.load(model_path))
    # Send the model to GPU/CPU
    model.to(device)
    # Set the model to evaluation mode
    model.eval()
    outputs = model(inputs)


if __name__ == '__main__':
    # Parse arguments to extract the path to the model file
    parser = ArgumentParser()
    parser.add_argument("model_path", help="path to the model file (.pt)")
    args = parser.parse_args()

    predict(args.model_path)
