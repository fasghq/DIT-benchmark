import numpy as np
import pandas as pd
import json
from tqdm import tqdm
from scipy.spatial.distance import pdist, squareform


def plinorder(prox, inperm=None, incoords=None, weights=None, max_iter=100):
    """
    Python implementation of unidimensional scaling using Guttman's updating algorithm with Pliner smoothing, with optional weights.
    
    Parameters:
    - prox: A symmetric proximity (dissimilarity) matrix with zero main diagonal.
    - incoords: An array of initial coordinates for the objects.
    - inperm: An input permutation of the first n integers.
    - weights: An optional weight matrix (must be the same shape as prox) that determines the importance of each proximity value.
    - max_iter: Maximum number of iterations for the Pliner smoothing procedure.

    Returns:
    - pcoordsort: Coordinates from the Pliner smoother ordered from most negative to most positive.
    - pperm: Object permutation indicating which objects are at which of the ordered coordinates in pcoordsort.
    - gcoordsort: Coordinates from Guttman update ordered from most negative to most positive.
    - gperm: Object permutation indicating which objects are at which of the ordered coordinates in gcoordsort.
    - gdiff: Value of the least-squares loss function for Guttman update.
    - pdiff: Value of the least-squares loss function for Pliner smoothing.
    """
    n = prox.shape[0]
    
    # gcoordprev = np.zeros(n)
    gcoord = np.zeros(n)

    # Initialize weight matrix
    if weights is None:
        weights = np.ones_like(prox)

    # Initial Guttman Coordinates based on input permutation
    if incoords is None:
        gpermprev = inperm.copy()
        gcoordprev = np.where(inperm == np.arange(n)[:, None])[1] # gcoordprev = np.where(inperm == np.arange(1, n + 1)[:, None])[1]
    else:
        gcoordprev = incoords.copy()
        gpermprev = np.argsort(gcoordprev) + 1

    # First Guttman update iteration (vectorized)
    gcoord = np.sum(weights * prox * np.sign(gcoordprev[:, None] - gcoordprev[None, :]), axis=1) / n

    gperm = np.arange(n) # gperm = np.arange(1, n + 1)
    gcoordsort = np.copy(gcoord)
    
    # Sorting coordinates and updating permutation
    gperm = gperm[np.argsort(gcoordsort)]
    gcoordsort = np.sort(gcoordsort)

    # Iterative Guttman Updates (vectorized)
    while not np.array_equal(gpermprev, gperm):
        gcoordprev = gcoord.copy()
        gpermprev = gperm.copy()
        gcoord = np.sum(weights * prox * np.sign(gcoordprev[:, None] - gcoordprev[None, :]), axis=1) / n

        # Sorting coordinates and updating permutation
        gperm = np.arange(n) # gperm = np.arange(1, n + 1)
        gcoordsort = np.copy(gcoord)
        gperm = gperm[np.argsort(gcoordsort)]
        gcoordsort = np.sort(gcoordsort)

    # Calculating Guttman loss (gdiff)
    gdiff = 0.5 * np.sum(weights * (prox - np.abs(gcoord[:, None] - gcoord[None, :])) ** 2)
    diss = squareform(pdist(gcoord.reshape((-1, 1)), 'euclidean'))
    gstress = np.sqrt(np.sum(weights * ((diss - prox) ** 2)) / np.sum(weights * (prox ** 2)))

    # Pliner smoothing
    ep = 2 * (np.max(np.sum(prox, axis=0)) / n)
    pperm = gperm.copy()
    pcoord = gcoord.copy()

    for k in range(2, max_iter + 2):
        ppermprev = np.random.permutation(n)

        while not np.array_equal(ppermprev, pperm):
            pcoordprev = pcoord.copy()
            ppermprev = pperm.copy()
            abst = np.abs(pcoordprev[:, None] - pcoordprev[None, :])
            factor = np.where(abst < ep, (pcoordprev[:, None] - pcoordprev[None, :]) / ep * (2 - (abst / ep)), np.sign(pcoordprev[:, None] - pcoordprev[None, :]))
            pcoord = np.sum(weights * prox * factor, axis=1) / n

            # Sorting coordinates and updating permutation
            pperm = np.arange(n) # pperm = np.arange(1, n + 1)
            pcoordsort = np.copy(pcoord)
            pperm = pperm[np.argsort(pcoordsort)]
            pcoordsort = np.sort(pcoordsort)

            # Calculating Pliner smoothing loss (pdiff)
            pdiff = 0.5 * np.sum(weights * (prox - np.abs(pcoord[:, None] - pcoord[None, :])) ** 2)

            diss = squareform(pdist(pcoord.reshape((-1, 1)), 'euclidean'))
            pstress = np.sqrt(np.sum(weights * ((diss - prox) ** 2)) / np.sum(weights * (prox ** 2)))

        # Update ep
        ep = ep * (max_iter - k + 1) / max_iter

    return pcoordsort, pperm, gcoordsort, gperm, gdiff, pdiff, gstress, pstress


collectionsDF = pd.read_csv('100NFTCollections - 100NFT_UPD.csv')
contractNames = collectionsDF['Symbol'].tolist()

rarity_meter = 'raritytools'
mode = 'train'

for contractName in tqdm(contractNames[2:]): # skip: ABS (1), 
    dissimilarities = np.load(f"dissimilarities_weights/{mode}/weighted_dissimilarities/{contractName}.npy")
    weights = np.load(f"dissimilarities_weights/{mode}/weights/{contractName}.npy")
    incoords = np.load(f"init_configurations/{mode}/{rarity_meter}/{contractName}.npy")
    try:
        pcoordsort, pperm, gcoordsort, gperm, gdiff, pdiff, gstress, pstress = plinorder(dissimilarities, incoords=incoords, inperm=None, weights=weights)
    except Exception as e:
        print(str(e))
        continue
    resultsDict = {
        'Pliner Coordinates Sorted': pcoordsort.tolist(), 
        'Guttman Coordinates Sorted': gcoordsort.tolist(),
        'Pliner Permutation': pperm.tolist(), 
        'Guttman Permutation': gperm.tolist(),
        'Pliner Loss': pdiff,
        'Guttman Loss': gdiff,
        'Pliner Stress': pstress,
        'Guttman Stress': gstress
    }
    with open(f'scalings/{mode}/{rarity_meter}/{contractName}.json', "w") as file:
        json.dump(resultsDict, file, indent=2) 
