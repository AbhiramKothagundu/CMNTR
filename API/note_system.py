import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import scipy.spatial as st
from time import gmtime, strftime
from ri import dsm, make_index, weight_func, remove_centroid
from inputProcesser.TenglishFormatter import process_user_input

class MultilingualNoteSystem:
    def __init__(self, notes_dir, embeddings_dir, dimension=300, nonzeros=8, delta=60):
        print("Loading multilingual BERT...")
        self.bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')
        self.bert_model = AutoModel.from_pretrained('bert-base-multilingual-cased')
        print("BERT model loaded successfully")
        
        # Directories
        self.notes_dir = notes_dir
        self.embeddings_dir = embeddings_dir
        
        # Random Indexing parameters
        self.dimension = dimension
        self.nonzeros = nonzeros
        self.delta = delta
        
        # Storage
        self.notes = {}  # Dictionary to store filename: content
        self.en_vocab = {}
        self.te_vocab = {}
        self.en_vectors = []
        self.te_vectors = []
        self.bert_embeddings = {}  # Dictionary to store filename: embedding
        
        # Load existing notes
        self._load_existing_notes()
        
    def _load_existing_notes(self):
        """Load existing notes from the notes directory"""
        for filename in os.listdir(self.notes_dir):
            if filename.endswith('.txt'):
                file_path = os.path.join(self.notes_dir, filename)
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    note_id = filename.replace('.txt', '')
                    self.process_note(content, note_id)
                    
    def process_note(self, text, note_id):
        """Process note using both BERT and Random Indexing"""
        # Process text through TenglishFormatter
        processed_text = process_user_input(text)
        
        # Store processed note
        self.notes[note_id] = processed_text
        
        # Create and store BERT embedding
        bert_embedding = self._create_bert_embedding(processed_text)
        self.bert_embeddings[note_id] = bert_embedding
        
        # Save embedding to file
        embedding_path = os.path.join(self.embeddings_dir, f"{note_id}_embedding.npy")
        np.save(embedding_path, bert_embedding.detach().numpy())
        
        # Split into English and Telugu words
        en_words, te_words = self._split_languages(processed_text)
        
        # Update Random Indexing vectors
        self._update_ri_vectors(en_words, te_words)
        
        return note_id
        
    def _split_languages(self, text):
        """Split text into English and Telugu words"""
        words = text.split()
        en_words = []
        te_words = []
        
        for word in words:
            if self._is_english(word):
                en_words.append(word.lower())
            else:
                te_words.append(word)
                
        return en_words, te_words
    
    def _is_english(self, word):
        """Check if word is English using ASCII range"""
        return all(ord(char) < 128 for char in word)
    
    def _update_ri_vectors(self, en_words, te_words):
        """Update Random Indexing vectors for both languages"""
        doc_vector = make_index(self.dimension, self.nonzeros)
        
        # Update English vectors
        for word in en_words:
            if word not in self.en_vocab:
                self.en_vocab[word] = [len(self.en_vectors), 1]
                self.en_vectors.append(np.zeros(self.dimension))
            else:
                self.en_vocab[word][1] += 1
                
            idx = self.en_vocab[word][0]
            weight = weight_func(self.en_vocab[word][1], len(self.en_vocab), self.delta)
            np.add.at(self.en_vectors[idx], doc_vector[:,0], doc_vector[:,1] * weight)
            
        # Update Telugu vectors
        for word in te_words:
            if word not in self.te_vocab:
                self.te_vocab[word] = [len(self.te_vectors), 1]
                self.te_vectors.append(np.zeros(self.dimension))
            else:
                self.te_vocab[word][1] += 1
                
            idx = self.te_vocab[word][0]
            weight = weight_func(self.te_vocab[word][1], len(self.te_vocab), self.delta)
            np.add.at(self.te_vectors[idx], doc_vector[:,0], doc_vector[:,1] * weight)
    
    def _create_bert_embedding(self, text):
        """Create BERT embedding for text"""
        inputs = self.bert_tokenizer(
            text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=512
        )
        
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
            
        return outputs.last_hidden_state.mean(dim=1)
    
    def search_notes(self, query, top_k=3):
        """Search notes using both BERT and Random Indexing"""
        # Process query through TenglishFormatter
        processed_query = process_user_input(query)
        
        # Create query embeddings
        query_bert = self._create_bert_embedding(processed_query)
        
        # Get query words
        en_words, te_words = self._split_languages(processed_query)
        
        # Calculate similarities
        similarities = []
        for note_id, note_text in self.notes.items():
            # BERT similarity
            bert_sim = torch.cosine_similarity(
                query_bert, 
                self.bert_embeddings[note_id], 
                dim=1
            ).mean().item()
            
            # RI similarity
            ri_sim = self._calculate_ri_similarity(en_words, te_words, note_text)
            
            # Combine scores
            combined_score = (bert_sim + ri_sim) / 2
            similarities.append((note_id, combined_score))
            
        # Sort and return top results
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for note_id, score in similarities[:top_k]:
            results.append({
                'note_id': note_id,
                'text': self.notes[note_id],
                'similarity': score
            })
            
        return results
    
    def _calculate_ri_similarity(self, query_en_words, query_te_words, note):
        """Calculate Random Indexing similarity"""
        note_en_words, note_te_words = self._split_languages(note)
        
        # Calculate English similarity
        en_sim = self._get_language_similarity(
            query_en_words,
            note_en_words,
            self.en_vocab,
            self.en_vectors
        )
        
        # Calculate Telugu similarity
        te_sim = self._get_language_similarity(
            query_te_words,
            note_te_words,
            self.te_vocab,
            self.te_vectors
        )
        
        return (en_sim + te_sim) / 2
    
    def _get_language_similarity(self, query_words, note_words, vocab, vectors):
        """Calculate similarity between word sets"""
        if not query_words or not note_words:
            return 0.0
            
        query_vector = np.zeros(self.dimension)
        note_vector = np.zeros(self.dimension)
        
        for word in query_words:
            if word in vocab:
                idx = vocab[word][0]
                query_vector += vectors[idx]
                
        for word in note_words:
            if word in vocab:
                idx = vocab[word][0]
                note_vector += vectors[idx]
                
        if np.any(query_vector) and np.any(note_vector):
            return 1 - st.distance.cosine(query_vector, note_vector)
        return 0.0