import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics.pairwise import manhattan_distances
from scipy.optimize import LinearConstraint, minimize
import json

def generatePermutations(n):
    return np.concatenate((np.eye(n, dtype=int), np.ones(n, dtype=int)[np.newaxis, :]), axis=0)


def stress_scaled(X, dissimilarities, alpha, weights):
    """
    Parameters
    ----------
    X : array-like of shape (n_samples, n_components)
        Configuration matrix. 

    dissimilarities : array-like of shape (n_samples, n_samples)
        Dissimilarity matrix.

    alpha : float
        Scaling factor for the distances between configuration points.

    weights: ndarray of shape (n_samples, n_samples)
        Weights for the stress calculation.

    Returns
    -------
    self : object
        Value of the normalized stress for given configuration and weights.
        Distances between the points of configuration are scaled by the
        precomputed factor that minimizes the stress.
    """
    distances = alpha * manhattan_distances(X)
    return np.sqrt(np.sum(weights * ((distances - dissimilarities) ** 2)) / np.sum(weights * (distances ** 2)))


def stress_fitted(omegas, X, dissimilarities, weights):
    """
    Parameters
    ----------
    X : array-like of shape (n_samples, n_components)
        Configuration matrix. 

    dissimilarities : array-like of shape (n_samples, n_samples)
        Dissimilarity matrix.

    omegas : ndarray of shape (n_components,)
        Coefficients for rarity calculatiom.

    weights: ndarray of shape (n_samples, n_samples)
        Weights for the stress calculation.

    Returns
    -------
    self : object
        Value of the normalized stress for given configuration, weights.
        Configuration is obtained depending on a coefficients set.
    """
    X = np.sum(X * omegas, axis=1).reshape(-1, 1)
    distances = manhattan_distances(X)
    return np.sqrt(np.sum(weights * ((distances - dissimilarities) ** 2)) / np.sum(weights * (distances ** 2)))


if __name__ == '__main__':

    collectionsDF = pd.read_csv(
        'C:\\Users\\Asus\\NFT_Rarity\ROAR\\100NFTCollections - 100NFT_UPD.csv',
        names=['Collection', 'Symbol', 'Address', 'Total Supply'],
        header=0,
        index_col=False
    )

    contractNames = collectionsDF['Symbol'].tolist()

    rarityMeters = ['roar']

    optimize = False

    for rarityMeter in tqdm(rarityMeters):
        for contractName in contractNames:

            if rarityMeter in ['raritytools', 'openrarity', 'nftgo']:
                X_init = np.load(f'C:\\Users\\Asus\\NFT_Rarity\\ROAR\\configurations\\{rarityMeter}\\{contractName}.npy').reshape(-1, 1)
                dissimilarities = np.load(f'C:\\Users\\Asus\\NFT_Rarity\\ROAR\\dissimilarities\\{contractName}.npy')
                weights = np.load(f'C:\\Users\\Asus\\NFT_Rarity\\ROAR\\weights\\{contractName}.npy')
                alpha = np.load(f'C:\\Users\\Asus\\NFT_Rarity\\ROAR\\alphas\\{rarityMeter}\\{contractName}_alpha.npy')

                stress = stress_scaled(X_init, dissimilarities, alpha, weights)
                with open(f'C:\\Users\\Asus\\NFT_Rarity\\ROAR\\results\\{rarityMeter}\\{contractName}_stress_noopt.npy', 'wb') as f:
                    np.save(f, stress)

            elif rarityMeter in ['kramer', 'roar']:
                X_init = np.load(f'C:\\Users\\Asus\\NFT_Rarity\\ROAR\\configurations\\{rarityMeter}\\{contractName}.npy')
                dissimilarities = np.load(f'C:\\Users\\Asus\\NFT_Rarity\\ROAR\\dissimilarities\\{contractName}.npy')
                weights = np.load(f'C:\\Users\\Asus\\NFT_Rarity\\ROAR\\weights\\{contractName}.npy')

                numberOfFreeCoefficients = X_init.shape[1]
                linearConstraints = []
                linearConstraints.append(generatePermutations(numberOfFreeCoefficients))
                linearConstraints.append(np.full(numberOfFreeCoefficients + 1, 0))
                linearConstraints.append(np.full(numberOfFreeCoefficients + 1, 1))
                linearConstraints = LinearConstraint(linearConstraints[0], linearConstraints[1], linearConstraints[2])
                coefficientsStart = np.full(numberOfFreeCoefficients, 1 / numberOfFreeCoefficients)

                minimizationResults = minimize(
                    stress_fitted, x0=coefficientsStart,
                    args=(X_init, dissimilarities, weights),
                    constraints=linearConstraints, options={'disp': True},
                )

                resultsDict = {
                    'x': minimizationResults.x.tolist(), 
                    'fun': minimizationResults.fun, 
                    'success': bool(minimizationResults.success),
                    'message': minimizationResults.message,
                    'jac': minimizationResults.jac.tolist(),
                    'nit': minimizationResults.nit,
                    'nfev': minimizationResults.nfev,
                    'njev': minimizationResults.njev,
                    'status': minimizationResults.status
                }

                with open(f'C:\\Users\\Asus\\NFT_Rarity\\ROAR\\results\\{rarityMeter}\\{contractName}.json', "w") as file:
                    json.dump(resultsDict, file, indent=2)
