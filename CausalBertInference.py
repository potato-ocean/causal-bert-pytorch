import pandas as pd
import torch
from CausalBert import CausalBertWrapper

import numpy as np
import pandas as pd 
from scipy.stats import norm, bernoulli


if __name__ == '__main__':
    cb = CausalBertWrapper()
    model = cb.model
    model.load_state_dict(torch.load('./causal-bert-peer-read-wrapper'))
    model.eval()
    print('reading csv')
    df = pd.read_csv('PeerRead_with_abstracts.zip')
    df['text'] = df['abstract']
    df['C'] = df['title_contains_deep'] | df['title_contains_neural'] | df['title_contains_embedding'] | df['title_contains_gan']
    df['C'] = df['C'].astype(int)
    df['T'] = df['num_ref_to_theorems'] > 0
    df['T'] = df['T'].astype(int)
    df['Y'] = df['accepted']

    ATE = cb.ATE(df['C'], df.text, platt_scaling=True)
    Q_ATT = cb.Q_ATT(df['C'], df['text'], df['T'])
    plug_in_ATT = cb.plug_in_ATT(df['C'], df['text'], df['T'], cb.loss_weights['g']) #multiplied by 10???
    print('done reading csv')    

    def simulation(z, b_1 = 5):
        pi_z = 0.07
        if z == 1:
            pi_z = 0.27
        else:
            pi_z = 0.07
        
        y1_prob = 1/(1 + np.exp(0.25 * 1 + b_1 * (pi_z - 0.2)))
        y0_prob = 1/(1 + np.exp(0.25 * 0 + b_1 * (pi_z - 0.2)))
        print("y1prob", y1_prob)
        print("y0prob", y0_prob)
        y1 = bernoulli.rvs(y1_prob)
        y0 = bernoulli.rvs(y0_prob)
        print("y1",y1)
        print("y0", y0)

        return y0, y1

    df['Y0'], df['Y1'] = df.apply(lambda row: simulation(row['C']), result_type="expand", axis=1)
    df_att = df[df['T'] == 1]

    ground_truth_att = np.mean(df_att['Y0'] - df_att['Y1'])
    print(df)
    print(df[df['Y0'] != 0])

    print("Peer Read data")
    print("ATE: ", ATE)
    print("Q_ATT: ", Q_ATT)
    print("plug_in_ATT: ", plug_in_ATT)
    print("ground_truth_att: ", ground_truth_att)

