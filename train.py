"""
Trains the model.
"""
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

from dataset import ImageAndKeypointDataset
from model import KeypointPredictionModel

DATA_PATH = './data/generated/'
MODEL_PATH = './model/model.pt'

TRAINING_SET_SIZE = 0.7
VALIDATION_SET_SIZE = 0.2
EPOCHS = 10
LEARNING_RATE = 1e-3
L2_PENALTY_VALUES = [0, 1e-4, 1e-3, 1e-2, 1e-1]
MINI_BATCH_SIZE = 16


def train(dataset, loss_fn, dev, weight_decay=0):
    # Initialize the model and send it to GPU/CPU
    model = KeypointPredictionModel().to(dev)
    # Data loader that returns randomized mini-batches
    dataloader = data.DataLoader(dataset, batch_size=MINI_BATCH_SIZE, shuffle=True, num_workers=2)
    # Model optimization method
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=weight_decay)

    # Set the model to training mode
    model.train()
    # Loop multiple epochs
    for epoch in range(EPOCHS):
        print(f"Training, epoch = {epoch}: ", end="")
        q = deque(maxlen=5)

        for i, datapoint in enumerate(dataloader):
            if i % 10 == 0:
                print(">", end="")

            inputs, labels = datapoint
            # Send the current mini-batch to the same device where the model is located
            inputs = inputs.to(dev)
            labels = labels.to(dev)
            # Forward: compute output values
            outputs = model(inputs)
            # Compute the loss and save the result
            loss = loss_fn(outputs, labels)
            # For debugging only
            # if i % 10 == 0:
            #     print(f"({loss.item():.2f})", end="")
            # Save the most recent loss values
            q.append(loss.item())
            # Backward: compute gradients
            loss.backward()
            # Update model parameters
            optimizer.step()
            # Reset gradients to zero
            optimizer.zero_grad()
        print(f" Finished. Loss for the last 5 mini-batches: {list(q)}")

    # Return the trained model
    return model


def validate_or_test(model, dataset, loss_fn, dev):
    # Data loader
    dataloader = data.DataLoader(dataset, batch_size=MINI_BATCH_SIZE, num_workers=2)
    # Total error
    total_error = 0.0
    # Set the model to evaluation mode
    model.eval()

    with torch.no_grad():
        for i, datapoint in enumerate(dataloader):
            if i % 10 == 0:
                print(">", end="")

            inputs, labels = datapoint
            inputs = inputs.to(dev)
            labels = labels.to(dev)

            outputs = model(inputs)
            loss = loss_fn(outputs, labels)

            batch_size = labels.size(0)
            total_error += loss.item() * batch_size
    avg_loss = total_error / len(dataset)
    print(f" Finished. Loss = {avg_loss:2f}")
    return avg_loss


def visualize_prediction(model, dataset, loss_fn, dev):
    print("Visualizing: ")
    dataloader = data.DataLoader(dataset)

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
        plt.show()
        pass


if __name__ == '__main__':
    # Use CUDA (GPU) if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # The full dataset from data files
    full_dataset = ImageAndKeypointDataset(DATA_PATH)
    # Split into training/validation/test sets
    train_set_size = int(len(full_dataset) * TRAINING_SET_SIZE)
    val_set_size = int(len(full_dataset) * VALIDATION_SET_SIZE)
    test_set_size = len(full_dataset) - train_set_size - val_set_size
    train_set, val_set, test_set = data.random_split(full_dataset, [train_set_size, val_set_size, test_set_size])

    # Loss function
    loss_function = nn.MSELoss()

    min_val_loss = None
    chosen_model = None
    chosen_wd = None
    # Try different L2 penalty values
    for wd in L2_PENALTY_VALUES:
        print(f"L2 penalty = {wd}")

        # Train a new model using the training set
        trained_model = train(train_set, loss_function, device, weight_decay=wd)

        # Calculate the loss using the validation set
        print("Validating: ", end="")
        val_loss = validate_or_test(trained_model, val_set, loss_function, device)

        # Save the model with the lowest validation loss
        if min_val_loss is None or val_loss < min_val_loss:
            min_val_loss = val_loss
            chosen_model = trained_model
            chosen_wd = wd
    # Calculate the loss again, this time using the test set
    print("Testing: ", end="")
    test_loss = validate_or_test(chosen_model, test_set, loss_function, device)

    # Visualize the prediction
    visualize_prediction(chosen_model, test_set, loss_function, device)

    # Save the chosen model to disk
    torch.save(chosen_model.state_dict(), MODEL_PATH)
