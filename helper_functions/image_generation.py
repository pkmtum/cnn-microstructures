import numpy as np
import scipy
import time

def getMeshgrid(n, x_end=1, y_end=1):
    """Gives Back a nxn regular Meshgrid for the midpoints of the nxn cells in the defined area of the I quadrant"""
    dx = x_end/n
    x = np.linspace(dx/2, x_end-dx/2, n)
    y = np.linspace(dx/2, y_end-dx/2, n)
    
    mesh = np.meshgrid(x, y)
    return mesh

    
def getDistances(xx, yy):
    x_cords = xx.flatten()  # 1D array of x_cords for all nodes
    y_cords = yy.flatten()  # 1D array of y_cords for all nodes
    # Now create square matrices (i,j) holding the x/y distance of nodes i & j
    xx_cords = np.tile(x_cords, (len(x_cords), 1))
    yy_cords = np.tile(y_cords, (len(y_cords), 1))
    xx_distances = np.abs(xx_cords - xx_cords.T)
    yy_distances = np.abs(yy_cords - yy_cords.T)

    return xx_distances, yy_distances


def nearestPD(A):
    """Find the nearest positive-definite matrix to input

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """

    B = (A + A.T) / 2
    _, s, V = np.linalg.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        return A3

    spacing = np.spacing(np.linalg.norm(A))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    I = np.eye(A.shape[0])
    k = 1
    while not isPD(A3):
        mineig = np.min(np.real(np.linalg.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3


def isPD(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = np.linalg.cholesky(B)
        return True
    except np.linalg.LinAlgError:
        return False


def createCovarianceMatrix(xx_distances, yy_distances, A_e, ar, sigma, rel_corr):
    """
    Creates the Covariance Matrix for use in a multivariate Gaussian RNG to
    create Images of blobs of 2 different phases on a 2D rectangle
    """
    lx = np.pi / (A_e * ar)
    ly = (np.pi * ar) / A_e
    d = np.log(1 / rel_corr)
    cov = (sigma ** 2) * np.exp(-d * (lx * xx_distances ** 2 + ly * yy_distances ** 2))

    eps = 1e-13
    offset = np.diag(np.repeat(eps,cov.shape[0]))

    # cov = cov * (1 - eps) + offset
    cov = cov + offset

    return cov


def createRandomValuesForImage(mean, cov):
    # approach 1: good but slow
    # ran = np.random.multivariate_normal(mean, cov)
    
    """
    # approach 2: fast but seems wrong
    l = cholesky(cov, check_finite=False, overwrite_a=True)
    ran = mean + l.dot(np.random.standard_normal(len(mean)))
    """

    # approach 3
    ran = mean + np.linalg.cholesky(cov) @ np.random.standard_normal(mean.size)

    return ran


def generateImageArray(ran, n, cutoff):
    img = np.zeros(len(ran))
    img[ran > cutoff] = 1
    return img


def generateImage(n=10, sigma=1, A_ellipse=0.25, a_r=1, rel_cor=0.01, vol_frac=0.5):
    # Global Parameters:
    n = n
    sigma = sigma
    A_ellipse = A_ellipse  # Area of the ellipse
    ar = a_r  # AxisRatio of the ellipse's axes rx/ry
    rel_cor = rel_cor  # relative value of correlation along ellipse boundary
    vol_frac = vol_frac  # fraction of the volume that will be filled with phase 1 in a mean sense
    z_cutoff = 0  # cutoff value fixed
    mu = z_cutoff - sigma * scipy.stats.norm.ppf(vol_frac)  # mean to achieve vol_frac of phase 1

    xx, yy = getMeshgrid(n)

    # start = time.process_time()
    xx_dist, yy_dist = getDistances(xx, yy)
    # print('Time for distances: ', time.process_time() - start)

    start = time.process_time()
    cov = createCovarianceMatrix(xx_dist, yy_dist, A_ellipse, ar, sigma, rel_cor)
    # print('Time for cov matrix: ', time.process_time() - start)

    mean = mu * np.ones(n ** 2)

    start = time.process_time()
    ran = createRandomValuesForImage(mean, cov)
    # print('Time for drawing from MVG: ', time.process_time() - start)

    # start = time.process_time()
    img = generateImageArray(ran, n, z_cutoff)
    # print('Time for image generation: ', time.process_time() - start)

    return ran, img


