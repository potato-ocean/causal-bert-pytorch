from scipy.spatial.distance import cosine
from transformers import BertTokenizer, BertModel
import torch
import pandas as pd
import logging

logger = logging.getLogger('my_logger')
logger.setLevel(logging.DEBUG)

# Create a file handler and set the level to DEBUG
file_handler = logging.FileHandler('topk.log')
file_handler.setLevel(logging.DEBUG)

# Create a formatter and set it for the file handler
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Add the file handler to the logger
logger.addHandler(file_handler)

original_buzzy = ['deep', 'neural', 'embed', 'adversarial net']

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') #replace with bert tokenizer
model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
    
def tokenize(text):
    marked_text = "[CLS] " + text + " [SEP]"
    tokens = tokenizer.tokenize(marked_text)
    return tokens

def get_embedding(text):
    tokens = tokenize(text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokens)
    segments_ids = [1] * len(indexed_tokens)

    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])
    model.eval()

    with torch.no_grad():
        outputs = model(tokens_tensor, segments_tensors)
        hidden_states = outputs[2]

    token_embeddings = torch.stack(hidden_states, dim=0)
    token_embeddings = torch.squeeze(token_embeddings, dim=1)
    # Swap dimensions 0 and 1.
    token_embeddings = token_embeddings.permute(1,0,2)
    token_vecs_cat = []

    for token in token_embeddings:
        # `token` is a [12 x 768] tensor

        # Concatenate the vectors (that is, append them together) from the last 
        # four layers.
        # Each layer vector is 768 values, so `cat_vec` is length 3,072.
        cat_vec = torch.cat((token[-1], token[-2], token[-3], token[-4]), dim=0)
        # Use `cat_vec` to represent `token`.
        token_vecs_cat.append(cat_vec)
    # Stores the token vectors, with shape [22 x 768]
    
    token_vecs_sum = []

    # `token_embeddings` is a [22 x 12 x 768] tensor.

    # For each token in the sentence...
    for token in token_embeddings:

        # Sum the vectors from the last four layers.
        sum_vec = torch.sum(token[-4:], dim=0)
        
        # Use `sum_vec` to represent `token`.
        token_vecs_sum.append(sum_vec)
    # For sentence embeds
    token_vecs = hidden_states[-2][0]
    sentence_embedding = torch.mean(token_vecs, dim=0)
    # token_embeddings = hidden_states[-2][0]
    # sentence_embedding = torch.mean(token_embeddings, dim=0)
    return tokens, token_vecs, sentence_embedding

original_buzzy_embeddings = {}
for w in original_buzzy:
    original_buzzy_embeddings[w] = get_embedding(w)[1]

df = pd.read_csv("PeerRead_with_abstract.csv")
def sim(abstract, original_buzzy_embeddings, cos, threshold=0.9):
    similar_words = {}
    tokens, token_vecs, _ = get_embedding(abstract)
    for token, word_embed in zip(tokens, token_vecs):
        for word, emb in original_buzzy_embeddings.items():
            similarity = 1 - cos(word_embed, emb[1, :])
            logger.info(f"{similarity}")
            if similarity > threshold: #and word not in original_buzzy:
                similar_words[token] = similarity
    return similar_words

cos = torch.nn.CosineSimilarity(dim=0)
logger.info("start top k")
L = df['abstract'].apply(lambda x: sim(x, original_buzzy_embeddings, cos))
similar_words = {}
for d in L:
    similar_words.update(d)
logger.info("stop top k")
file_path = "similar_words.txt"
with open(file_path, 'w') as file:
    for k, v in similar_words.items():
        file.write(f'{k} : {v}' + '\n')
