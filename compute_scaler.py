import numpy as np
import pandas as pd
from tqdm import tqdm
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

    for rarityMeter in tqdm(rarityMeters[:1]):
        for contractName in contractNames[:1]:
            X_init = np.load(f'C:\\Users\\Asus\\NFT_Rarity\\ROAR\\configurations\\{rarityMeter}\\{contractName}.npy').reshape(-1, 1)
            dissimilarities = np.load(f'C:\\Users\\Asus\\NFT_Rarity\\ROAR\\dissimilarities\\{contractName}.npy')
            weights = np.load(f'C:\\Users\\Asus\\NFT_Rarity\\ROAR\\weights\\{contractName}.npy')

            alpha = np.sum(weights * (dissimilarities ** 2)) / np.sum(weights * dissimilarities * manhattan_distances(X_init))

            with open(f'C:\\Users\\Asus\\NFT_Rarity\\ROAR\\alphas\\{rarityMeter}\\{contractName}_alpha.npy', 'wb') as f:
                np.save(f, alpha)

            
            

            

            
            
            
