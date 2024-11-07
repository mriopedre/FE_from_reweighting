#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def calc_center_of_bins(bins):
    """Centers the bins for plotting the histogram correctly"""
    centered_bins = (bins[:-1] + bins[1:]) / 2
    return centered_bins

def calculate_weights(KBT, bias):
    """Calculates the weight of each frame based on the bias

    Args:
        KBT (float): value of kT (2.57 for 298K)
        bias (np.array): bias of each frame. Obtained from the metadynamics simulation

    Returns:
        np.array: weight of each frame
    """
    weights = np.exp(bias / KBT)
    weights /= weights.sum()
    return weights

def plot_reweighted_FE_1D(cv, weights, sum_hills = None, plot_histogram = None):
    """Plot the rewighted free energy of a 1D CV and the probability distribution.
    If sum Hills is provided, also plot the FE from sum_hills for comparison.

    Args:
        cv (np.array): CV values for the CV of interest
        weights (np.array): Array of weights for each frame
        sum_hills (str, optional): File path to the results from plumed sum_hills. Defaults to None.
        plot_histogram (bool, optional): If True, plot the raw histogram. Defaults to None.
    """
    #  Make a histogram of the values of the CV weigthed by the free energy of the frame
    hist, bins = np.histogram(cv, bins=50, weights=weights, density=True)

    # Calculate the bin centers to plot the histogram correctly
    bins = calc_center_of_bins(bins)
    # Calculate the free energy from the histogram
    fe = - np.log(hist) * KBT
    fe = fe - fe.min()

    # Plot the free energy 
    fig,axes=plt.subplots(nrows=2, ncols=1,figsize=(6,6),sharex=True)
    axes[0].plot(bins, fe, label = 'Reweighted FE')

    # Plot the FE from sum_hills for comparison if the file is provided
    if sum_hills is not None:
        extract_sum_hills(sum_hills, bins, axes)

    if plot_histogram is not None:
        plot_raw_histogram(cv, axes)

    # Plot the histogram
    axes[0].set_title('Free Energy')
    axes[0].set_ylabel('Free Energy (kJ/mol)')
    axes[1].plot(bins, hist, color = 'red')
    axes[1].set_title('Probability')
    axes[1].set_ylabel('Probability from the histogram)')
    axes[1].set_xlabel('CV value')
    plt.tight_layout()
    plt.show()

def plot_raw_histogram(cv, axes):
    """Plot the raw histogram of the CV values"""
    hist_raw, bins_raw = np.histogram(cv, bins=50, density=True)
    bins_raw = calc_center_of_bins(bins_raw)
    ax_hist = axes[1].twinx()  # Create secondary axis
    ax_hist.plot(bins_raw, hist_raw, color='green')
    ax_hist.set_ylabel('Raw histogram Density')

def extract_sum_hills(sum_hills, bins, axes):
    """Extract the free energy from sum_hills and plot it"""
    try:
        cv_, fe_, _ = np.loadtxt(sum_hills).T
    except ValueError:
        cv_, fe_,= np.loadtxt(sum_hills).T
    axes[0].scatter(cv_, fe_, color = 'orange', label='sum_hills')
    axes[0].legend()
    axes[0].set_xlim(bins.min(), bins.max())

def plot_reweighted_FE_2D(cv1, cv2, weights, vmax = 80):
    """Plot the free energy of a 2D CV

    Args:
        cv1 (np.array): CV values for the CV of interest
        cv2 (np.array): CV values for the CV of interest
        weights (np.array): Array of weights for each frame
        vmax (int, optional): Maximum value of free energy shown. Defaults to 80.
    """

    #Calculate the 2D Histogram
    #Density normalizes it to form a probability density.
    bins, cv1, cv2 = np.histogram2d(cv1, cv2, bins=70, weights=weights, density=True)

    # Calculate the bin centers to plot the histogram correctly
    cv1 = calc_center_of_bins(cv1)
    cv2 = calc_center_of_bins(cv2)

    # Calculate the free energy from the histogram. 
    # Set the inf values caused by points without samples to a randomly big number
    fe = - np.log(bins) * KBT
    fe[fe == np.inf] = 1000000

    # Plot the free energy
    fig, ax = plt.subplots(1,1, figsize=(5,5), dpi=100)

    #Calculate the aspect ratio and plot
    aspect_ratio = abs((cv1.max() - cv1.min()) / (cv2.max() - cv2.min()))
    im = ax.imshow(fe.T, extent=(cv1.min(), cv1.max(), cv2.min(), cv2.max()), 
                   vmax=vmax, cmap=cm.RdBu_r, origin = 'lower', aspect = aspect_ratio)
    ax.set_xlabel('CV1')
    ax.set_ylabel('CV2')

    # Add a colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.05, 0.7])
    cbl = fig.colorbar(im, cax=cbar_ax)
    cbl.ax.set_ylabel("kJ/mol" )
    plt.show()

#%% Example usage
# Input folders and files
# File is the colvar from plumed Driver - see example Driver
# Set KBT according to the temperature of the simulation
fol = '/example_folder/'
file = fol+'/COLVAR_REWEIGHT'
KBT=2.57

# Read the colvar file and split it adequatelly 
# Example here for a given Colvar file. Split accordingly to the field in your Colvar.
time, qz, qy, qx, bias = np.loadtxt(file).T

# Calculate weights of frames using the bias
weights = calculate_weights(KBT, bias)

# Examples to plot the free energy for 1D or 2D CVs
# sum_hills - optional, file - point to file with sum_hills results. Shows comparison.
# plot_histogram - optional, bolean - shows sampling of the CV by printing the unweigthed histogram. 
plot_reweighted_FE_1D(qz, weights, sum_hills = f+'/qz.dat', plot_histogram=True)
# Plot 2D CVs
plot_reweighted_FE_2D(qz, qx, weights)
