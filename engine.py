import torch
import matplotlib.pyplot as plt
import os

def train_model(model: torch.nn.Module,
                Xtrain: torch.Tensor,
                ytrain: torch.Tensor,
                Xtest: torch.Tensor,
                ytest: torch.Tensor,
                epochs: int,
                savedir: str,
                filename: str):
    """
    Trains the PyTorch model

    Parameters:
        model (nn.Module): model to be trained
        Xtrain (torch.Tensor): training features
        ytrain (torch.Tensor): training labels
        Xtest (torch.Tensor): testing features
        ytest (torch.Tensor): testing labels
        epochs (int): number of epochs
        savedir (str): path to save model
        filename (str): filename of saved model
    """

    # Set up loss function and optimizer
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=0.5e-4, eps = 1e-3, lr = 0.001)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Set up the lists to store optimization metrics
    train_loss_list, test_loss_list = [], []

    # Variable to store best loss
    best_loss = float('inf')
    # Create save folder if it does not exist
    os.makedirs(savedir, exist_ok=True)

    # Loop over epochs
    for epoch in range(epochs):
        ### Training step
        model.train()

        # Forward pass
        output = model(Xtrain).squeeze()
        # Compute loss
        loss = loss_fn(output, ytrain)
        train_loss_list.append(loss.item())

        ### Test step
        model.eval()
        # Forward pass
        with torch.inference_mode():
            output = model(Xtest).squeeze()
        # Compute loss
        test_loss = loss_fn(output, ytest)
        test_loss_list.append(test_loss.item())

        # PyTorch necessities
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Save best model
        if test_loss.item() < best_loss:
            best_loss = test_loss.item()
            torch.save(model.state_dict(), os.path.join(savedir, f"{filename}.pth"))

        if epoch % 500 == 0:
            print(f"epoch: {epoch}")
            print(f"Train loss: {train_loss_list[-1]:.3f}, test loss: {test_loss_list[-1]:.3f}")

    plt.plot(train_loss_list, label="train")
    plt.plot(test_loss_list, label="test")
    plt.legend()
    plt.yscale("log")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.show()


    # Save the model
    torch.save(model.state_dict(), os.path.join(savedir, f"{filename}.pth"))

    return train_loss_list, test_loss_list
