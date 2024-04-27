import numpy as np
import pandas as pd
import json
from tqdm import tqdm
from multiprocessing import Pool


def salesJsonToNP(tradedataJson):
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


def kernelEpanechnikov(blockNumber1, blockNumber2, blocksPerOneDay=6000, dayLimit=7):
    blockNumberLimit = dayLimit * blocksPerOneDay
    return 3 / 4 * (np.power(1 - np.abs(blockNumber1 - blockNumber2) / blockNumberLimit, 2))


def init_worker(sales_dict):
    global sales_per_id_dict
    sales_per_id_dict = sales_dict


def loop_for_i(id1, id2):
    global sales_per_id_dict
    salePairs = np.array([[np.hstack([s1, s2]) for s1 in sales_per_id_dict[id1]] for s2 in sales_per_id_dict[id2]]).reshape(-1, 4)
    delta_i_j = 0
    w_i_j = 0
    for salePair in salePairs:
        k = kernelEpanechnikov(salePair[0], salePair[2])
        w_i_j += k
        delta_i_j += k * np.abs(np.log(salePair[1] / salePair[3]))
    if w_i_j != 0:
        delta_i_j /= w_i_j
    else: 
        delta_i_j = 0
    return [delta_i_j, w_i_j]
    #return w_i_j


def getDissimilarity(sales, fname_d, fname_w):
    ids = sorted(np.unique(sales[:, 0]))
    length = len(ids)
    
    sales_per_id = {}
    for i in range(length):
        sales_per_id[i] = sales[sales[:, 0] == ids[i]][:, 1:]

    items = []
    for i in range(length):
        for j in range(i + 1, length): 
            items.append((i, j))

    with Pool(initializer=init_worker, initargs=(sales_per_id,)) as pool:
        results = pool.starmap(loop_for_i, items)
    
    del sales_per_id

    dissimilarities = np.zeros((length, length), dtype=np.float64)
    for id, r in enumerate(results):
        dissimilarities[items[id][0], items[id][1]] = r[0]
        dissimilarities[items[id][1], items[id][0]] = r[0]
    with open(fname_d, 'wb') as f:
        np.save(f, dissimilarities)
    del dissimilarities

    weights = np.zeros((length, length), dtype=np.float64)
    for id, r in enumerate(results):
        weights[items[id][0], items[id][1]] = r[1]
        weights[items[id][1], items[id][0]] = r[1]
    with open(fname_w, 'wb') as f:
        np.save(f, weights)
    del weights

    # weights = np.zeros((length, length), dtype=np.float64)
    # for id, r in enumerate(results):
    #     weights[items[id][0], items[id][1]] = r
    #     weights[items[id][1], items[id][0]] = r
    # with open(fname_w, 'wb') as f:
    #     np.save(f, weights)
    # del weights


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

    for contractName in tqdm(contractNames[1:]): #15
        with open(f'{tradedataPath}\\{contractName}.json', "r") as file:
            sales = salesJsonToNP(json.load(file))

        # print(contractName)

        getDissimilarity(sales, f'C:\\Users\\Asus\\NFT_Rarity\\ROAR\\dissimilarities\\{contractName}.npy', f'C:\\Users\\Asus\\NFT_Rarity\\ROAR\\weights\\{contractName}.npy')
        
        
