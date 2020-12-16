"""
Visualize the results calculated by a given model on an image.
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data

from dataset import ImageAndKeypointDataset
from model import KeypointPredictionModel

DATA_PATH = './data/generated/'
MODEL_FILE = './model/model_final.pt'


def visualize_prediction(model, dataset, loss_fn, dev):
    print("Visualizing: ")
    dataloader = data.DataLoader(dataset, shuffle=True)

    model.eval()
    with torch.no_grad():
        # Randomly pick a data point
        input, label = next(iter(dataloader))

        # Predict
        input = input.to(dev)
        label = label.to(dev)
        output = model(input)
        loss = loss_fn(output, label)

        print(f"Loss = {loss.item():.2f}")

        # Convert tensors to NumPy arrays
        input = input.cpu().numpy()
        label = label.cpu().numpy()
        output = output.detach().cpu().numpy()
        img = np.squeeze(input, axis=0).transpose((1, 2, 0))
        kps_label = np.squeeze(label, axis=0).reshape(-1, 2)
        kps = np.squeeze(output, axis=0).reshape(-1, 2)

        # Draw
        plt.imshow(img)
        plt.scatter(kps_label[:, 0], kps_label[:, 1], s=3, c="y")
        plt.scatter(kps[:, 0], kps[:, 1], s=5, c="r")
        plt.axis('off')
        plt.show()


if __name__ == '__main__':
    # Use CUDA (GPU) if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Dataset
    dataset = ImageAndKeypointDataset(DATA_PATH)
    # Load the model
    model = KeypointPredictionModel()
    model.load_state_dict(torch.load(MODEL_FILE))
    model.to(device)
    # Loss function
    loss_function = nn.MSELoss()

    # Visualize the prediction
    visualize_prediction(model, dataset, loss_function, device)
