import torch
from mlp import MLP
from data_loader import ais_dataset
from scripts.data_loader import DataLoader

def infer(points1,points2):
    #model =  load model
    # emb1 = model(points1)
    # emb2 = model(points2)
    # d = torch.matmul(emb1, emb2.t())
    #idx1,idx2= hungarian algor
    #points1[idx1] -> points2[idx2]
    pass

def data_loaded():
    # Define date and time filter
    date_key = '03-11-2022'

    # PATHS, dataframe and shpfile #
    # Define paths
    base_path = "C:\\Users\\abelt\\OneDrive\\Desktop\\Kandidat\\"
    ## File names ##
    # AIS
    ais_files = {
        #'02-11-2022': 'ais\\ais_110215.csv',
        '03-11-2022': 'ais\\ais_110315.csv',
        #'05-11-2022': 'ais\\ais_1105.csv'
    }
    # SAR
    sar_files = {
        '02-11-2022': 'sar\\Sentinel_1_detection_20221102T1519.json',
        '03-11-2022': 'sar\\Sentinel_1_detection_20221103T154515.json',
        '05-11-2022': 'sar\\Sentinel_1_detection_20221105T162459.json'
    }
    # Norsat
    norsat_files = {
        '02-11-2022': 'norsat\\Norsat3-N1-JSON-Message-DK-2022-11-02T151459Z.json',
        '03-11-2022': 'norsat\\Norsat3-N1-JSON-Message-DK-2022-11-03T152759Z.json',
        '05-11-2022': 'norsat\\Norsat3-N1-JSON-Message-DK-2022-11-05T155259Z.json'
    }

    # LOADING #
    data_loader = DataLoader(base_path = base_path, ais_files = ais_files, date_key = date_key)
    ais_loader, sar_loader, norsat_loader = data_loader.load_data()

    ######### SAR #########
    # images by date_key: sar_loader.dfs_sar
    # objects by date_key: sar_loader.sar_object_dfs
    ######### AIS #########
    #ais_loader.dfs_ais
    ######### Norsat #########
    #norsat_loader.dfs_norsat

    #sar_data = sar_loader.sar_object_dfs[date_key].copy()
    ais_data = ais_loader.dfs_ais[date_key].copy()
    #norsat_data = norsat_loader.dfs_norsat[date_key].copy()
    return ais_data

def train():
    #data = data_loaded().groupby('mmsi')
    #print(data.groups.keys())
    EMBED_DIM = 4
    model = torch.nn.Sequential(
        MLP(4, 128, 100),
        MLP(100, 128, EMBED_DIM),
    )

    dataset = ais_dataset(None)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=3, shuffle=True)

    for ais in dataloader:
        print(ais.size())
        d = model.forward(ais[0],ais[1])
        d = torch.softmax(d, dim=1)
        diag = torch.diag(d)
        loss = -torch.log(diag).mean()
        print(loss)                
        break
    

if __name__ == '__main__':
    train()