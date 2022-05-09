def local_linear_regression(x, I, dx=0.1, kernel_width=1):
    """At each point, use a local linear regression to estimate the value
    
    Arguments:
        x, I: Unsorted coordinates of the data points
        dx: spacing of the points at which to estimate the regression curve
        kernel_width: width (standard deviation) of the Gaussian smoothing kernel
        
    Return Value:
    xs, Is: X and I coordinates of the regression estimator.  xs will have
    a spacing of dx, and runs from the min to the max of the input x array.
    
    Details:
    This is essentially a dumb implementation of the formula given on page 50
    of Bowman & Azzalini (1997), "Applied smoothing techniques for data analysis".
    """
    regression_x = np.arange(x.min(), x.max(), dx)
    # Construct a 2D array representing the difference in X coordinate between
    # the points and the estimator positions
    displacements = x[:, np.newaxis] - regression_x[np.newaxis, :]
    weights = np.exp(-displacements**2 / (kernel_width**2 * 2))
    # Compute the weighted means at each estimate position
    #local_means = np.sum(weights * I[:, np.newaxis], axis=0) / np.sum(weights, axis=0)
    # Compute the weighted moments for each estimate position
    s0 = np.mean(weights, axis=0)[np.newaxis, :]
    s1 = np.mean(weights * displacements**1, axis=0)[np.newaxis, :]
    s2 = np.mean(weights * displacements**2, axis=0)[np.newaxis, :]
    local_linear_estimator = np.mean(
        ((s2 - s1*displacements)*weights*I[:,np.newaxis])
        /
        (s2*s0 - s1**2),
        axis=0
    )
    return regression_x, local_linear_estimator