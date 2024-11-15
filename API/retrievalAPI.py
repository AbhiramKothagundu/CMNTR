from dotenv import load_dotenv
import os
import torch
import numpy as np
from transformers import BertTokenizer, BertModel
from note_system import MultilingualNoteSystem
from sklearn.metrics.pairwise import cosine_similarity

import sys
sys.path.append(os.path.abspath('../API/inputProcesser'))


# Now import the function
from TenglishFormatter import process_user_input


# Set random seed for reproducibility
random_seed = 42
torch.manual_seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(random_seed)

# Initialize the multilingual tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = BertModel.from_pretrained('bert-base-multilingual-cased')

load_dotenv()
NOTES_DIRECTORY = os.getenv("NOTES_DIRECTORY")
EMBEDDINGS_DIRECTORY = os.getenv("EMBEDDINGS_DIRECTORY")


note_system = MultilingualNoteSystem(NOTES_DIRECTORY, EMBEDDINGS_DIRECTORY)


# Function to compute the embedding of a document or query
def compute_embedding(text):
    # Tokenize and encode the text
    encoding = tokenizer.batch_encode_plus(
        [text],
        padding=True,
        truncation=True,
        return_tensors='pt',
        add_special_tokens=True
    )

    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']

    # Generate embeddings
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state.mean(dim=1)  # Average pooling along sequence length

    return embeddings.numpy()

# Function to load embeddings from the .npy files
def load_embeddings():
    embeddings = {}
    
    # Iterate through all the files in the embeddings directory
    for filename in os.listdir(EMBEDDINGS_DIRECTORY):
        if filename.endswith('_embedding.npy'):
            file_path = os.path.join(EMBEDDINGS_DIRECTORY, filename)
            document_name = filename.replace('_embedding.npy', '')
            
            # Load the embedding and store it
            embeddings[document_name] = np.load(file_path)

    return embeddings

def find(query):
    """Find similar notes using MultilingualNoteSystem"""
    search_results = note_system.search_notes(query)
    
    # Format results for CLI
    results = []
    for result in search_results:
        results.append((
            result['note_id'],
            result['similarity'],
            result['text']
        ))
    
    return results

# Example usage
if __name__ == "__main__":
    query_text = "Ravi doctor అవ్వాలని అనుకున్నాడు, కానీ Arun?"
    results = find(query_text)
    
    # Print the top 3 documents
    print(f"\nTop 3 results for the query: {query_text}")
    for idx, (doc_name, similarity, content) in enumerate(results):
        print(f"\nRank {idx + 1}:\nDocument: {doc_name}\nSimilarity: {similarity}\nContent:\n{content}\n")
