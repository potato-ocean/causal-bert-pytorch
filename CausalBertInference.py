import pandas as pd
import torch
from CausalBert import CausalBertWrapper

import numpy as np
import pandas as pd 
from scipy.stats import norm, bernoulli



def compute_model_dependent_stats(df):
    cb = CausalBertWrapper()
    model = cb.model
    model.load_state_dict(torch.load('./causal-bert-peer-read-wrapper'))
    model.eval()
    

    ATE = cb.ATE(df['C'], df.text, platt_scaling=True)
    Q_ATT = cb.Q_ATT(df['C'], df['text'], df['T'])
    plug_in_ATT = cb.plug_in_ATT(df['C'], df['text'], df['T'], cb.loss_weights['g']) #multiplied by 10???
    

    


    print("Peer Read data")
    print("ATE: ", ATE)
    print("Q_ATT: ", Q_ATT)
    print("plug_in_ATT: ", plug_in_ATT)


def compute_true_propensity_scores(df):
    propensity_score_0 = df[df['C'] == 1]
    propensity_score_1 = df[df['C'] == 0]




def compute_model_independent_stats(df):
    def simulation_y0(z, b_1 = 5):
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

        return y0
    
    def simulation_y1(z, b_1 = 5):
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

        return y1


    df['Y0'] = df['C'].apply((lambda x: simulation_y0(x)))
    df['Y1'] = df['C'].apply((lambda x: simulation_y1(x)))

    assert df['Y0'].sum() != 0
    assert df['Y1'].sum() != df['Y1'].__len__
    print(df)
    df_att = df[df['T'] == 1]
    ground_truth_att = np.mean(df_att['Y0'] - df_att['Y1'])
    print("ground_truth_att: ", ground_truth_att)



if __name__ == '__main__':
    print('reading csv')
    df = pd.read_csv('PeerRead_with_abstracts.zip')
    df['text'] = df['abstract']
    df['C'] = df['title_contains_deep'] | df['title_contains_neural'] | df['title_contains_embedding'] | df['title_contains_gan']
    df['C'] = df['C'].astype(int)
    df['T'] = df['num_ref_to_theorems'] > 0
    df['T'] = df['T'].astype(int)
    df['Y'] = df['accepted']
    print('done reading csv')
    #compute_model_dependent_stats(df)
    compute_model_independent_stats(df)    