from CausalBert import CausalBertWrapper
import torch


cb = torch.load(CausalBertWrapper, ',/causal-bert-peer-read')
ATE = cb.ATE(df['C'], df.text, platt_scaling=True)
Q_ATT = cb.Q_ATT(df['C'], df['text'], df['T'])
plug_in_ATT = cb.plug_in_ATT(df['C'], df['text'], df['T'], cb.loss_weights['g'])
print("Peer Read data")
print("ATE: ", ATE)
print("Q_ATT: ", Q_ATT)
print("plug_in_ATT: ", plug_in_ATT)