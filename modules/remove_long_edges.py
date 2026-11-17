# file remove_long_edges.py contains useful functions to remove long edges from tetrahedras
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt

"""Methods :
- find_best_breakpoint : find the best breakpoint in the dataset"""

def find_best_breakpoint(lengths, export : bool = False, verbose = False):
    """Find the best breakpoint in the dataset by fitting 2 negative slope lines.
    Args:
        lengths(list): List of edges lengths, want to find a bkp in this distrib.
        verbose (bool): wether to plot the edges distribution and best rupture point.
    Return:
        cut_links (int) : a threshold to remove lengths forming cavities."""
    
    counts, bin_edges = np.histogram(lengths, bins=50)  # histogramme
    x = bin_edges[:-1].reshape(-1,1)  # classes centers, reshape for sklearn
    y = counts          # frequencies

    # Smoothing the function
    f_c = 0.25 # Cutted frequency, highes = smallest sigma window
    dx = bin_edges[1]-bin_edges[0]
    sigma_samples = 0.1326 / (f_c * dx) # Width of Gaussian window, expressed in number of histogram bins. 0.1326 = ln(2)/2pi i.e. a -3dB filter.
    y_smooth = gaussian_filter1d(y, sigma=sigma_samples, mode = 'reflect')

    if verbose:
        plt.plot(x,y_smooth , label='smoothed histogram', color='blue')

    #Initialization
    best_rss = np.inf
    best_bkp = None

    # Iterate over all possible breakpoints
    for i in range(5, len(x)): # Avoid extreme points
        # Split the data into two segments
        x_left, y_left = x[:i] , y_smooth[:i]
        x_right, y_right = x[i:], y_smooth[i:]

        # Fit linear models to each segment
        model_left = LinearRegression().fit(x_left, y_left)
        model_right = LinearRegression().fit(x_right, y_right)
        
        # Calculate residual sum of squares (RSS)
        y_pred_left = model_left.predict(x_left)
        y_pred_right = model_right.predict(x_right)
        
        rss = np.sum((y_left - y_pred_left)**2) + np.sum((y_right - y_pred_right)**2)
        
        # Update best breakpoint if current RSS is lower
        if rss < best_rss:
            best_rss = rss
            best_bkp = i
        
    # Calculating the threshold
    cut_links = x[best_bkp][0]

    # --- Once the best break is found, fit again on both segments ---
    x_left, y_left = x[:best_bkp], y_smooth[:best_bkp]
    x_right, y_right = x[best_bkp:], y_smooth[best_bkp:]

    model_left = LinearRegression().fit(x_left, y_left)
    model_right = LinearRegression().fit(x_right, y_right)

    y_pred_left = model_left.predict(x_left)
    y_pred_right = model_right.predict(x_right)

    # --- Plot the two best fitted lines ---
    if verbose:
        plt.plot(x_left, y_pred_left, color='red', label='best fit left')
        plt.plot(x_right, y_pred_right, color='green', label='best fit right')
        plt.axvline(cut_links, color='k', linestyle='--', label=f'breakpoint={cut_links:.2f}')
        plt.legend()
        plt.xlabel('Length')
        plt.ylabel('Occurancies')
        plt.title('Smoothed histogram with best piecewise linear fit')


        print(f"All tetras withs edges longer than {cut_links} has been removed.")
        plt.show()

    if export:
        # Flatten x because sklearn/scipy arrays are often shaped (n, 1)
        x_flat = x.ravel()

        # Name of exported files
        base_name = f"breakpoint_{cut_links:.4f}".replace(".", "p")

        # 1) Courbe lissée : x, y_smooth
        np.savetxt(
            f"{base_name}_y_smooth.txt",
            np.column_stack((x_flat, y_smooth)),
            header="x y_smooth",
            comments=""
        )

        # 2) Left adjustement : x_left, y_pred_left
        np.savetxt(
            f"{base_name}_y_pred_left.txt",
            np.column_stack((x_left.ravel(), y_pred_left)),
            header="x y_pred_left",
            comments=""
        )

        # 3) Right adjustement : x_right, y_pred_right
        np.savetxt(
            f"{base_name}_y_pred_right.txt",
            np.column_stack((x_right.ravel(), y_pred_right)),
            header="x y_pred_right",
            comments=""
        )

        # 4) Breakpoint : just a file with bkp value
        with open(f"{base_name}_cut_links.txt", "w") as f:
            f.write("cut_links\n")
            f.write(f"{cut_links:.8f}\n")

    return cut_links