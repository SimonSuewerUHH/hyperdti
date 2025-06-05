import pandas as pd
import torch

from data.graph_gen import HypergraphDataGenerator
from helper.parser import parse_args
from helper.train import train


def main():
    cfg = parse_args()
    torch.manual_seed(cfg.seed)

    df = pd.read_csv('dataset/BindindDB/full_data.csv')
    df_5 = df[0:400]
    smiles = df_5['SMILES']
    proteins = df_5['Proteins']
    sequences = df_5['sequence']

    # 1) Synthetic Data erzeugen
    generator = HypergraphDataGenerator(
        drug_smiles_list=smiles,
        proteins=proteins,
        sequences=sequences
    )
    saved_path = 'dataset/BindindDB/test1.pkl'
    if generator.data_exists(saved_path):
        print(f"Loading data from {saved_path}")
        data = generator.load_data(saved_path)
    else:
        print(f"Generating data and saving to {saved_path}")
        # Generate the data
        # Note: This will take a while
        data = generator.generate()
        generator.save_data(data, saved_path)

    train(data, cfg)


if __name__ == '__main__':
    main()
