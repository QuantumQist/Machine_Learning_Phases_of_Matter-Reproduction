import numpy as np
import torch
from pathlib import Path

def get_data(chain_length: int,
             directory: str,
             device: torch.device,
             label_dtype: torch.dtype = torch.float,
             samples_per_temp_train: int = 2500,
             samples_per_temp_test: int = 250):
    """
    Imports training and testing data from the given directory for specified chain length.

    Parameters:
        chain_length (int): The length of the side of the lattice considered.
            Denoted by "L" in the paper.
            Total system size is L^2.
        directory (str): The path to the directory where the data is stored.
        device (torch.device): The device on which the data should be stored.
        label_dtype (torch.dtype, optional): The data type for the labels. Defaults to torch.float.
        samples_per_temp_train (int, optional): Number of samples per temperature in the training set. Defaults to 2500.
        samples_per_temp_test (int, optional): Number of samples per temperature in the testing set. Defaults to 250.

    Returns:
        tuple: (
            Xtrain (torch.Tensor),                 # training features on device
            Xtest (torch.Tensor),                  # testing features on device
            ytrain (torch.Tensor),                 # training labels on device (dtype = label_dtype)
            ytest (torch.Tensor),                  # testing labels on device (dtype = label_dtype)
            training_temperatures (np.ndarray),    # per-sample temperatures in the training set
            testing_temperatures (np.ndarray)      # per-sample temperatures in the test set
        )
    """
    def get_features(file_path: Path):
        """
        Helper function for getting features from a given directory.

        Parameters:
            file_path (Path): The path to the directory where the data is stored.

        Returns a torch tensor with the data.
        """
        data = np.loadtxt(file_path, dtype=int)
        return torch.tensor(data, dtype=torch.float)

    def get_labels(file_path: Path):
        """
        Helper function for getting labels from a given directory.

        Parameters:
            file_path (Path): The path to the directory where the data is stored.

        Returns a torch tensor with the data.
        """
        labels = np.loadtxt(file_path)
        return torch.tensor(labels, dtype=label_dtype)

    ### Training features
    file_path = Path(directory) / f"L_{chain_length}" / "Xtrain.txt"
    Xtrain = get_features(file_path).to(device=device)

    ### Testing features
    file_path = Path(directory) / f"L_{chain_length}" / "Xtest.txt"
    Xtest = get_features(file_path).to(device=device)

    ### Training labels
    file_path = Path(directory) / f"L_{chain_length}" / "ytrain.txt"
    ytrain = get_labels(file_path).to(device=device)

    ### Testing labels
    file_path = Path(directory) / f"L_{chain_length}" / "ytest.txt"
    ytest = get_labels(file_path).to(device=device)

    ### Temperatures in the training and testing sets
    temperature_string = '1.0000000000000000 1.0634592657106510 1.1269185314213019 1.1903777971319529 1.2538370628426039 1.3172963285532548 1.3807555942639058 1.4442148599745568 1.5076741256852078 1.5711333913958587 1.6345926571065097 1.6980519228171607 1.7615111885278116 1.8249704542384626 1.8884297199491136 1.9518889856597645 2.0153482513704155 2.0788075170810667 2.1422667827917179 2.2057260485023691 2.2691853142130203 2.3326445799236715 2.3961038456343227 2.4595631113449739 2.5230223770556250 2.5864816427662762 2.6499409084769274 2.7134001741875786 2.7768594398982298 2.8403187056088810 2.9037779713195322 2.9672372370301834 3.0306965027408346 3.0941557684514858 3.1576150341621370 3.2210742998727881 3.2845335655834393 3.3479928312940905 3.4114520970047417 3.4749113627153929 3.5383706284260401'
    unique_temperatures = np.array(temperature_string.split(), dtype=float)
    training_temperatures = np.repeat(unique_temperatures, samples_per_temp_train)
    testing_temperatures = np.repeat(unique_temperatures, samples_per_temp_test)

    return Xtrain, Xtest, ytrain, ytest, training_temperatures, testing_temperatures