from transformers import BertTokenizer, BertModel
import torch
import pandas as pd
import logging
import numpy as np

from transformers import DistilBertTokenizer
from transformers import DistilBertModel

logging.basicConfig(filemode='w')
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

deep = "moreover , unlike the original nade , our training procedure scales to deep models ."

neural = "scalability properties of deep neural networks raise key research questions , particularly as the problems considered become larger and more challenging ."

embed = "deep learning embeddings have been successfully used for many natural language processing ( nlp ) problems ."

adversarial = "we introduce a class of cnns called deep convolutional generative adversarial networks ( dcgans ) , that have certain architectural constraints , and demonstrate that they are a strong candidate for unsupervised learning ."

contextualized_buzzy = [deep, neural, embed, adversarial]

tokenizer = BertTokenizer.from_pretrained('distilbert-base-uncased') #replace with bert tokenizer
model = BertModel.from_pretrained('fine-tuned-causal-bert', output_hidden_states=True)
    
def tokenize(text):
    '''
    Tokenize a given piece of text. Helper function for get_embedding.
    Args:
        text (str): Some text
    Returns:
        tokens (list): Tokenized text by BERT
    '''
    marked_text = "[CLS] " + text + " [SEP]"
    tokens = tokenizer.tokenize(marked_text)
    return tokens

def get_embedding(text):
    '''
    Computes the embedding for a piece of text.
    Args:
        text (str): Some text
    Returns:
        tokens (list): The tokens generated from the tokenizer
        token_vecs (list): The embedding of the token
        abstract_embedding (torch.Tensor): The abstract level embedding
    '''
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
    token_embeddings = token_embeddings.permute(1,0,2)
    # Now N x 12 x 768
    token_vecs_cat = []

    for token in token_embeddings:
        # concatenate last 4 layers since shows best performance
        # See https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/
        cat_vec = torch.cat((token[-1], token[-2], token[-3], token[-4]), dim=0)
        token_vecs_cat.append(cat_vec)
    token_vecs_sum = []
    for token in token_embeddings:

        # now sum the last 4 layers
        sum_vec = torch.sum(token[-4:], dim=0)
        token_vecs_sum.append(sum_vec)

    # Compute the embeddings for the entire abstract
    token_vecs = hidden_states[-2][0]
    abstract_embedding = torch.mean(token_vecs, dim=0)
    return tokens, token_vecs, abstract_embedding

original_buzzy_embeddings = {}
# must parse some buzzy words different based off BERT tokenizer
HOLD_AD = []
HOLD_EMB = []
for part in contextualized_buzzy:
    e = get_embedding(part)
    for token, token_vec in zip(e[0], e[1]):
        if token in ["ad", "##vers", "##aria", "##l]"]:
            HOLD_AD.append(token_vec)
        if token in ["em", "##bed", "##ding", "##s"]:
            HOLD_EMB.append(token_vec)
        if token in ['deep', 'embedding', 'adversarial', 'neural']:
            original_buzzy_embeddings[token] = get_embedding(token)[1][1, :] # rid of [cls] and [sep]

# Stack and sum the sub-word token embeddings
original_buzzy_embeddings["adversarial"] = torch.stack(HOLD_AD, dim=0).sum(dim=0).squeeze()
original_buzzy_embeddings["embed"] = torch.stack(HOLD_EMB, dim=0).sum(dim=0).squeeze()

# read the abstracts
def sim(abstract, original_buzzy_embeddings, cos, threshold=0.3):
    '''
    Args:
        abstract (string): The abstract
        original_buzzy_embeddings: A dictionary of the original buzzy embeddings
                                    and their embeddings
        cos (torch.nn): Pytorch cosine similarity function
        threshold (float): Value to threshold similarity scores by for efficiency

    Returns:
        similar_word (dictionary): A dictionary of similar words with keys = tokens
                                    and values = similarity score
    '''
    similar_words = {}
    tokens, token_vecs, _ = get_embedding(abstract)
    for token, word_embed in zip(tokens, token_vecs):
        for word, emb in original_buzzy_embeddings.items():
            # Compare the similarity between each word and the words in the
            # buzzy set
            # 1 cos means values close to 0 are similar.
            similarity = 1 - cos(word_embed, emb)
            if similarity < threshold:
                similar_words[token] = similarity
    return similar_words

df = pd.read_csv("PeerRead_with_abstract.csv")
cos = torch.nn.CosineSimilarity(dim=0)
logger.info("start top k")
# Apply to all abstracts in dataset
L = df.head['abstract'].apply(lambda x: sim(x, original_buzzy_embeddings, cos))
similar_words = {}
# update the dictionary with our list of similar word dictionaries
for d in L:
    similar_words.update(d)
logger.info("stop top k")
file_path = "similar_words_new.txt"
# write the words and similarity scores to file
with open(file_path, 'w') as file:
    for k, v in similar_words.items():
        file.write(f'{k} : {v}' + '\n')