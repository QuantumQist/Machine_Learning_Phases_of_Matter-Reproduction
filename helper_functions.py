import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

def plot_output_in_plane(model, X, temperatures):
    """
    Plots the output layer in style of Fig. 6 in the paper.

    Parameters:
        model (nn.Module): model used for inference.
        X (torch.Tensor): features used for inference.
        test_temperatures (torch.Tensor): temperatures used for inference.
    """
    # Prepare everything for computation
    model.eval()

    # Forward pass
    with torch.inference_mode():
        output = model(X).cpu()

    output = torch.permute(output, dims = (1, 0)).numpy()


    # Center value
    center_val = 2 / np.log(1 + np.sqrt(2))

    # Define colormap normalization with specified center
    norm = TwoSlopeNorm(vmin=np.min(temperatures),
                        vcenter=center_val,
                        vmax=np.max(temperatures))
    plot = plt.scatter(output[0], output[1], alpha = 0.1, c = temperatures, cmap='bwr', norm=norm, s = 8, edgecolors='black')
    plt.xlabel("Neuron 1")
    plt.ylabel("Neuron 2")

    # Add colorbar
    cbar = plt.colorbar(plot)
    cbar.solids.set_alpha(1.)
    cbar.set_label('Temperature')

    plt.show()

def make_predictions(model, X):
    """
    Makes predictions using a trained model.

    Parameters:
        model (nn.Module): The model used for inference.
        X (torch.Tensor): The input data. Each entry is a single spin system.

    Returns (torch.Tensor): The predictions.
    """
    # Prepare everything for calculations.
    model.eval()

    # Perform inference
    with torch.inference_mode():
        output = model(X)

    # Pick the entry with highest probability
    preds = torch.argmax(output, -1)

    return preds.squeeze()

def compute_accuracy(model, X, y):
    """
    Computes model accuracy.

    Parameters:
        model (nn.Module): model to compute accuracy on.
        X (torch.Tensor): input features
        y (torch.Tensor): output labels
    """
    model.eval()

    # Get predictions
    preds = make_predictions(model, X)
    # Check the number of correct elements
    agreement = (preds == y).sum().item()
    # return the fraction of correct counts
    return agreement / len(X)

def plot_accuracy_vs_temperature(model, X, y, temperatures, samples_per_temperature, make_plot = True):
    """
    Plots model accuracy vs. temperature.

    Parameters:
        model (nn.Module): model to compute accuracy on.
        X (torch.Tensor): input features
        y (torch.Tensor): output labels
        temperatures (torch.Tensor): temperatures used for computation
        samples_per_temperature (int): number of X & y samples per temperature in the data
        make_plot (bool): whether to make a plot or not.

    Returns:
        the list of accuracies for each temperature.
    """
    # Prepare everything for computation
    model.eval()

    # List to store accuracy on
    accuracy_list = []

    # Loop over unique temperatures
    for idx in range(len(temperatures)):
        # Pick subset corresponding to a given temperature
        idx_min = idx * samples_per_temperature
        idx_max = idx_min + samples_per_temperature
        # Get predictions
        accuracy = compute_accuracy(model, X[idx_min:idx_max], y[idx_min:idx_max])
        accuracy_list.append(accuracy)

    if make_plot:
        plt.plot(temperatures, accuracy_list, "-o")
        plt.xlabel("Temperature")
        plt.ylabel("Accuracy")
        plt.axvline(x = 2 / np.log(1 + np.sqrt(2)), color = "orange")
        plt.ylim(-0.05, 1.05)
        plt.show()

    return accuracy_list

def plot_setup(model, X, y, idx = None):
    """
    Plots a setup with the model prediction.

    Parameters:
        model (nn.Module): model used for inference.
        X (torch.Tensor):  features
        y (torch.Tensor):  labels
        idx (int): index of the image used for plotting.
            If not provided, random setup is plotted.
    """
    # If idx not provided, plot a random image
    if idx is None:
        idx = np.random.randint(X.shape[0])

    # Get model prediction
    pred = make_predictions(model, X[idx]).item()

    agreement = (pred == y[idx])

    X_numpy = X[idx].cpu().numpy().reshape( (L, L) )

    plt.imshow(X_numpy, vmin=0., vmax = 1., cmap="gray_r")

    phase = "ferromagnetic" if pred == 0 else "paramagnetic"

    color = "green" if agreement else "red"
    plt.title(f"predicted phase: {phase}, color=color")

def plot_output_layer(model, X , temperatures , samples_per_temp = 250, make_plot = True):
    """
    Plots the output layer as a function of temperature.

    Parameters:
        model (nn.Module): model used for inference.
        X (torch.Tensor): features
        temperatures (torch.Tensor): temperatures describing the model
        samples_per_temp (int): number of samples per temperature. Default is 250 - value in testing set.
        make_plot (bool): whether to make a plot or not.

    Returns a tuple with 4 lists
        - means for neuron 1
        - error in mean estimation for neuron 1
        - means for neuron 2
        - error in mean estimation for neuron 2
    """
    # Put everything on "cpu" for convenience
    model.eval()

    # Create lists to store the results
    means_neuron1, stds_neuron1 = [], []
    means_neuron2, stds_neuron2 = [], []

    for i in range(len(temperatures)):
        idx_min = i * samples_per_temp
        idx_max = idx_min + samples_per_temp

        # Forward pass
        with torch.inference_mode():
            output = model(X[idx_min:idx_max])
        # Softmax
        probs = torch.permute(output, (1,0))

        # Get means
        means = torch.mean(probs, dim = -1)
        means_neuron1.append(means[0].item())
        means_neuron2.append(means[1].item())
        # Get sample standard deviation
        # Chack sample standard deviation vs. population standard deviation
        # https://www.statology.org/population-vs-sample-standard-deviation/
        stds = torch.std(probs, dim = -1, unbiased=False)
        stds_neuron1.append(stds[0].item()/(samples_per_temp**0.5))
        stds_neuron2.append(stds[1].item()/(samples_per_temp**0.5))


    if make_plot:
        plt.errorbar(temperatures, means_neuron1, yerr=stds_neuron1, fmt="-o", markersize = 2,
                     color = "blue", label = "Neuron 1", capsize=2)
        plt.errorbar(temperatures, means_neuron2, yerr=stds_neuron2, fmt="-o", markersize = 2,
                     color = "red", label = "Neuron 2", capsize=2)
        plt.xlabel("Temperature")
        plt.ylabel("Average output layer")
        plt.axvline(x = 2 / np.log(1 + np.sqrt(2)), color = "orange", label = "Critical temperature")
        plt.legend()
        plt.xlim(temperatures[0], temperatures[-1])
        plt.ylim(bottom=-.05, top = 1.05)
        plt.show()

    return means_neuron1, stds_neuron1, means_neuron2, stds_neuron2

