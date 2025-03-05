import numpy as np

def decadalInternalVariability(simulations, gmt=None, window_size=10):
    """
    Calculate the decadal internal variability.

    Parameters:
        simulations (array-like): 4D array of model simulations with dimensions (ensemble_members, years, latitudes, longitudes).
        gmt (array-like): Global mean temperature (optional).
        window_size (int): Size of the moving average window.

    Returns:
        DIVs (array-like): Decadal internal variability with dimensions (ensemble_members, years, latitudes).
    """
    if gmt is None:
        # Calculate global mean temperature
        gmt = simulations.mean(axis=(0, 2, 3))
    # Calculate internal variability for each ensemble member
    IVs = np.array([gmt - simulations[i].mean(axis=(1, 2)) for i in range(simulations.shape[0])])
    # Smooth the internal variability using a moving average
    DIVs = np.array([np.convolve(IV, np.ones(window_size) / window_size, mode='same') for IV in IVs])
    return DIVs

def regionalMeanTemperature(simulations, n_lat=2, n_lon=2, window_size=10):
    """
    Calculate the regional mean temperature and its smoothed internal variability.

    Parameters:
        simulations (array-like): 4D array of model simulations with dimensions (ensemble_members, years, latitudes, longitudes).
        n_lat (int): Number of latitude divisions.
        n_lon (int): Number of longitude divisions.
        window_size (int): Size of the moving average window.

    Returns:
        rmt (array-like): Regional mean temperature with dimensions (latitudes, longitudes).
        smoothed_riv (array-like): Smoothed regional internal variability with dimensions (latitudes, longitudes).
    """
    shape = simulations.shape
    # Reshape simulations to divide the globe into regions
    tas_regional = simulations.reshape(shape[0], shape[1], n_lat, shape[2] // n_lat, n_lon, shape[3] // n_lon)
    # Calculate regional mean temperature
    rmt = tas_regional.mean(axis=(0, 3, 5))
    # Calculate regional internal variability
    riv = rmt - tas_regional.mean(axis=(3, 5))
    # Create a 1D convolution kernel for the moving average
    kernel = np.ones(window_size) / window_size
    # Smooth the regional internal variability using a moving average
    smoothed_riv = np.apply_along_axis(lambda x: np.convolve(x, kernel, mode='same'), axis=1, arr=riv)
    return rmt, smoothed_riv

def extract_data(data, model, n_lat_riv=None, n_lon_riv=None, max_run=8, anchor='div', ncp=1, window_size=None):
    """
    Extract data for model training.

    Parameters:
        data (dict): Dictionary containing model data.
        model (str): Name of the model.
        n_lat_riv (int): Number of latitude divisions for regional internal variability.
        n_lon_riv (int): Number of longitude divisions for regional internal variability.
        max_run (int): Maximum number of ensemble members to consider.
        anchor (str): Anchoring method ('div', 'riv', or 'iv_pca').
        ncp (int): Number of components for IV-PCA.
        window_size (int): Size of the moving average window.

    Returns:
        A (array-like): Features for training.
        X (array-like): Input data (tas).
        Y (array-like): Target data (rmt).
    """
    shape = data[model]['tas'].shape
    shape_rmt = data[model]['rmt'].shape
    
    # Select a random subset of ensemble members
    idx_runs = np.random.randint(0, shape[0], max_run)
    if max_run is not None:
        # Select subset of data based on the chosen ensemble members
        Y = np.array([data[model]['rmt'] for i in range(shape[0])])[idx_runs, :, :, :].reshape(min(shape[0], max_run) * shape_rmt[0], shape_rmt[1] * shape_rmt[2])
        X = data[model]['tas'][idx_runs, :, :, :].reshape(min(shape[0], max_run) * shape[1], shape[2] * shape[3])
        if anchor == 'riv':
            A = data[model]['riv'][idx_runs, :, :, :].reshape(min(shape[0], max_run) * shape[1], n_lat_riv * n_lon_riv)
        elif anchor == 'div':
            A = data[model]['div'][idx_runs, :].reshape(min(shape[0], max_run) * shape[1], 1)
        elif anchor == 'iv_pca':
            A = data[model]['iv_pca'][idx_runs, :, :].reshape(min(shape[0], max_run) * shape[1], ncp)
        
        if window_size is not None:
            # Create a 1D convolution kernel for the moving average
            kernel = np.ones(window_size) / window_size
            # Apply the moving average along axis=1
            Y = np.apply_along_axis(lambda x: np.convolve(x, kernel, mode='same'), axis=1, arr=Y)
    else:
        # Use all ensemble members
        Y = np.array([data[model]['rmt'] for i in range(shape[0])]).reshape(shape[0] * shape_rmt[0], shape_rmt[1] * shape_rmt[2])
        X = data[model]['tas'].reshape(shape[0] * shape[1], shape[2] * shape[3])
        A = data[model]['riv'].reshape(shape[0] * shape[1], n_lat_riv * n_lon_riv)
        if anchor == 'riv':
            A = data[model]['riv'][idx_runs, :, :, :].reshape(min(shape[0], max_run) * shape[1], n_lat_riv * n_lon_riv)
        elif anchor == 'div':
            A = data[model]['div'][idx_runs, :].reshape(min(shape[0], max_run) * shape[1], 1)
        elif anchor == 'iv_pca':
            A = data[model]['iv_pca'][idx_runs, :, :].reshape(min(shape[0], max_run) * shape[1], ncp)
    return A, X, Y
