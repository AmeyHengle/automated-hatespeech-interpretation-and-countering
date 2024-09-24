import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"

print(f"Generating Embeddings")

df = pd.read_csv('data/facts.csv')
# df['fact'] = df['fact'].apply(lambda x: x[0] if isinstance(x, list) else x)

# Step 2: Set device to CUDA if available, otherwise CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Step 3: Load a pre-trained model from sentence-transformers and move it to the appropriate device
model = SentenceTransformer('all-MiniLM-L6-v2').to(device)

# Step 4: Compute embeddings for the facts
fact_embeddings = model.encode(df['fact'].tolist(), convert_to_tensor=True, device=device)

def find_most_similar_fact(statement):
    # Step 5: Compute embedding for the input statement and move to the appropriate device
    statement_embedding = model.encode(statement, convert_to_tensor=True, device=device)

    # Step 6: Calculate cosine similarities
    cosine_scores = util.pytorch_cos_sim(statement_embedding, fact_embeddings)[0]

    # Get the index of the most similar fact
    most_similar_idx = cosine_scores.argmax().item()
    
    # Return the most similar fact and its score
    return df.iloc[most_similar_idx], cosine_scores[most_similar_idx].item()
