import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn import manifold
from sklearn.metrics.pairwise import manhattan_distances


scoresPaths = {
    'raritytools': 'C:\\Users\\Asus\\NFT_Rarity\\ROAR\\rarity_scores\\raitytools_scores',
    'kramer': 'C:\\Users\\Asus\\NFT_Rarity\ROAR\\rarity_scores\\kramer_scores',
    'openrarity': 'C:\\Users\\Asus\\NFT_Rarity\\ROAR\\rarity_scores\\openrarity_scores',
    'nftgo': 'C:\\Users\\Asus\\NFT_Rarity\\ROAR\\rarity_scores\\nftgo_scores'
}

if __name__ == '__main__':

    collectionsDF = pd.read_csv(
        'C:\\Users\\Asus\\NFT_Rarity\ROAR\\100NFTCollections - 100NFT_UPD.csv',
        names=['Collection', 'Symbol', 'Address', 'Total Supply'],
        header=0,
        index_col=False
    ).drop(0, axis='index')

    contractNames = collectionsDF['Symbol'].tolist()

    rarityMeters = ['raritytools', 'kramer', 'openrarity', 'nftgo']

    optimize = False

    for rarityMeter in tqdm(rarityMeters[:1]):
        for contractName in contractNames[:1]:
            X_init = np.load(f'C:\\Users\\Asus\\NFT_Rarity\\ROAR\\configurations\\{rarityMeter}\\{contractName}.npy').reshape(-1, 1)
            dissimilarities = np.load(f'C:\\Users\\Asus\\NFT_Rarity\\ROAR\\dissimilarities\\{contractName}.npy')
            weights = np.load(f'C:\\Users\\Asus\\NFT_Rarity\\ROAR\\weights\\{contractName}.npy')
            alpha = np.load(f'C:\\Users\\Asus\\NFT_Rarity\\ROAR\\alphas\\{rarityMeter}\\{contractName}_alpha.npy')

            if optimize == True:
                mds = manifold.MDS(
                    n_components=1,
                    metric=False, 
                    n_init=1,
                    max_iter=3000, 
                    eps=1e-9, 
                    dissimilarity="precomputed", 
                    n_jobs=1,
                )
                pos = mds.fit(dissimilarities, init=X_init, weight=weights).embedding_
                stress = mds.stress_
                print(stress)
                with open(f'C:\\Users\\Asus\\NFT_Rarity\\ROAR\\results\\{rarityMeter}\\{contractName}_pos.npy', 'wb') as f:
                    np.save(f, pos)
                with open(f'C:\\Users\\Asus\\NFT_Rarity\\ROAR\\results\\{rarityMeter}\\{contractName}_stress.npy', 'wb') as f:
                    np.save(f, stress)
            else:
                distances = alpha * manhattan_distances(X_init)
                stress = np.sqrt(np.sum(weights * ((distances - dissimilarities) ** 2)) / np.sum(weights * (distances ** 2)))
                print(stress)
                with open(f'C:\\Users\\Asus\\NFT_Rarity\\ROAR\\results\\{rarityMeter}\\{contractName}_stress_noopt.npy', 'wb') as f:
                    np.save(f, stress)
            

            

            
            
            
