from scipy.spatial.distance import cosine
from transformers import BertTokenizer, BertModel
import torch

original_buzzy = ['deep', 'neural', 'embed', 'adversarial net']

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') #different model
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

    token_embeddings = hidden_states[-2][0]
    sentence_embedding = torch.mean(token_embeddings, dim=0)
    return sentence_embedding

original_buzzy_embeddings = {}
for w in original_buzzy:
  original_buzzy_embeddings[w] = get_embedding(w)

df['tokens'] = df['abstract'].apply(tokenize)
print(df['tokens'])
tokens_df = pd.DataFrame()


def sim(tokens, original_buzzy_embeddings, threshold=0.7):
  similar_words = []
  print("start")
  for token in tokens:
    token_embedding = get_embedding(token)
    for word, emb in original_buzzy_embeddings.items():
      similarity = 1 - cosine(token_embedding, emb)
      if similarity > threshold:
        similar_words.append(word)
  print("done")
  return similar_words

results = set(df['tokens'].apply(lambda x: sim(x, original_buzzy_embeddings)))
