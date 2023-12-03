import pandas as pd
import torch
from CausalBert import CausalBertWrapper

if __name__ == '__main__':
    cb = CausalBertWrapper()
    model = cb.model
    model.load_state_dict(torch.load('./causal-bert-peer-read-wrapper'))
    model.eval()

    df = pd.read_csv('PeerRead_with_abstracts.zip')
    df['text'] = df['abstract']
    df['C'] = df['title_contains_deep'] | df['title_contains_neural'] | df['title_contains_embedding'] | df['title_contains_gan']
    df['C'] = df['C'].astype(int)
    df['T'] = df['num_ref_to_theorems'] > 0
    df['T'] = df['T'].astype(int)
    df['Y'] = df['accepted']

    ATE = cb.ATE(df['C'], df.text, platt_scaling=True)
    Q_ATT = cb.Q_ATT(df['C'], df['text'], df['T'])
    plug_in_ATT = cb.plug_in_ATT(df['C'], df['text'], df['T'], cb.loss_weights['g'])
    print("Peer Read data")
    print("ATE: ", ATE)
    print("Q_ATT: ", Q_ATT)
    print("plug_in_ATT: ", plug_in_ATT)