"""
Trains the model.
"""
import copy
from argparse import ArgumentParser, FileType
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

from dataset import ImageAndKeypointDataset
from model import KeypointPredictionModel

DEFAULT_DATASET_PATH = './data/generated/'
DEFAULT_MODEL_PATH = './model/model.pt'

EPOCHS = 100
LEARNING_RATE = 1e-4
MOMENTUM = 0
MINIBATCH_SIZE = 16


def train(dataset_path=DEFAULT_DATASET_PATH, model_path=DEFAULT_MODEL_PATH):
    # Use CUDA (GPU) if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Initialize the model and send it to GPU/CPU
    model = KeypointPredictionModel().to(device)
    # Load the data
    dataset = ImageAndKeypointDataset(dataset_path)
    # Data loader that returns randomized minibatches
    data_loader = data.DataLoader(dataset, batch_size=MINIBATCH_SIZE, shuffle=True, num_workers=2)
    # Model optimization method
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # Loss function
    loss_fn = nn.MSELoss()

    # Set the model to training mode
    model.train()
    # Loop multiple epochs
    for epoch in range(EPOCHS):
        print(f"\nEpoch #{epoch} running loss values: ")
        for inputs, labels in data_loader:
            # Send the current minibatch to the same device where the model is located
            inputs = inputs.to(device)
            labels = labels.to(device)
            # Forward: compute output values
            outputs = model(inputs)
            # Compute the loss and save the result
            loss = loss_fn(outputs, labels)
            print(f"{round(loss.item(), 2)} ", end="")
            # Backward: compute gradients
            loss.backward()
            # Update model parameters
            optimizer.step()
            # Reset gradients to zero
            optimizer.zero_grad()

    # Save the trained model to a file
    trained_model = copy.deepcopy(model.state_dict())
    torch.save(trained_model, model_path)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-d", "--data", type=FileType("r"), help="path to the data file")
    parser.add_argument("-m", "--model", help="path to which the model will be saved")
    args = parser.parse_args()

    train()
