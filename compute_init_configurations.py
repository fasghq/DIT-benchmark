import numpy as np
import pandas as pd
import json
from tqdm import tqdm


def salesJsonToNPorDF(tradedataJson):
    sales = []
    for trade in tradedataJson:
        sale = [int(trade['tokenId']), int(trade['blockNumber'])]
        price = 0
        for fee in ['sellerFee', 'protocolFee', 'royaltyFee']:
            if trade[fee]:
                price += float(trade[fee]['amount'])
        sale.append(price)
        sales.append(sale)
    sales = np.array(sales)
    sales = sales[sales[:, 2] != 0]
    return sales


tradedataPath = 'C:\\Users\\Asus\\NFT_Rarity\\ROAR\\tradedata'

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

    for rarityMeter in tqdm(rarityMeters):
        for contractName in contractNames:
            with open(f'{tradedataPath}\\{contractName}.json', "r") as file:
                sales = salesJsonToNPorDF(json.load(file))
            traded_ids = sorted(np.unique(sales[:, 0]))
            traded_ids = [int(id) for id in traded_ids]
            scoresDF = pd.read_csv(f'{scoresPaths[rarityMeter]}/{contractName}_{rarityMeter}_scores.csv', index_col=0)

            X_init = np.zeros(len(traded_ids), dtype=float)
            for i, id in enumerate(traded_ids):
                X_init[i] = scoresDF.loc[scoresDF['Token Id'] == id, 'Rarity score'].iloc[0]
            
            with open(f'C:\\Users\\Asus\\NFT_Rarity\\ROAR\\configurations\\{rarityMeter}\\{contractName}.npy', 'wb') as f:
                np.save(f, X_init)

            scored_ids = scoresDF['Token Id'].unique()
            untraded_ids = [int(id) for id in scored_ids if id not in traded_ids]
            ids = {'traded': traded_ids, 'untraded': untraded_ids}
            with open(f'C:\\Users\\Asus\\NFT_Rarity\\ROAR\\ids\\{rarityMeter}\\{contractName}.json', 'w') as f:
                json.dump(ids, f, indent=4)
            
            
