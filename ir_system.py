import re
import time
import os
import csv
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import PorterStemmer


CORPUS_FILE = 'Articles.csv' 


STEMMER = PorterStemmer()

CORPUS = []
DOCUMENT_TITLES = []

def load_corpus_from_csv(filepath):
    print(f"Loading documents from CSV file: {filepath}")
    local_corpus = []
    local_titles = []
    
    title_index = 0
    content_index = 0 
    
    if not os.path.exists(filepath):
       
        print(f"Error: File '{filepath}' not found. Please ensure it is in the correct directory.")
        return [], []

    try:
        
        with open(filepath, 'r', encoding='latin-1', newline='') as csvfile:
            reader = csv.reader(csvfile)
            
            
            next(reader) 
            
            for row in reader:
                
                if len(row) > max(title_index, content_index):
                    local_titles.append(row[title_index])
                    local_corpus.append(row[content_index])
                    
    except Exception as e:
        print(f"Could not read CSV file {filepath}. Please check the file format and ensure  the Title (Index {title_index}) and Content (Index {content_index}) are correctly configured.")
        print(f"Detailed error: {e}")
        return [], []
                
    print(f"Successfully loaded {len(local_corpus)} documents.")
    return local_corpus, local_titles

# --- Core IR System Class ---

class TFIDF_IR_System:
   
    def __init__(self, corpus, titles):
        self.corpus = corpus
        self.titles = titles
        self.vectorizer = None
        self.doc_vectors = None

    def preprocess(self, text):
        text = re.sub(r'<.*?>', ' ', text)
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)
        tokens = text.split()
        stemmed_tokens = [STEMMER.stem(token) for token in tokens]
        return ' '.join(stemmed_tokens)

    def build_index(self):
        print("--- Indexing Phase Started ---")
        
        if not self.corpus:
            print("Error: Corpus is empty. Cannot build index.")
            return

        
        processed_corpus = [self.preprocess(doc) for doc in self.corpus]
        
       
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            norm='l2',
            ngram_range=(1, 2) 
        )
        
        start_time = time.time()
        self.doc_vectors = self.vectorizer.fit_transform(processed_corpus)
        indexing_time = time.time() - start_time
        
        print(f"Total documents indexed: {len(self.corpus)}")
        print(f"Vocabulary size (including N-grams): {len(self.vectorizer.get_feature_names_out())}")
        print(f"Indexing completed in {indexing_time:.4f} seconds.")
        print("------------------------------")
        
    def retrieve(self, query, k=3):
        if self.doc_vectors is None:
            print("Error: Index not built. Call build_index() first.")
            return []

       
        processed_query = self.preprocess(query)
        query_vector = self.vectorizer.transform([processed_query])
        
        if query_vector.sum() == 0:
            return "No valid terms found in query after preprocessing."

        target_term = STEMMER.stem('system') 
        weight_factor = 5
        
        
        if target_term in self.vectorizer.vocabulary_:
            term_index = self.vectorizer.vocabulary_[target_term]
            
            
            if query_vector[0, term_index] > 0:
                query_vector[0, term_index] *= weight_factor
                print(f"DEBUG: Manually boosted weight for term '{target_term}' by factor of {weight_factor}.")


        start_time = time.time()
        similarity_scores = cosine_similarity(query_vector, self.doc_vectors).flatten()
        query_speed = time.time() - start_time
        
        ranked_indices = np.argsort(similarity_scores)[::-1]
        
        results = []
        for rank, doc_index in enumerate(ranked_indices[:k]):
            score = similarity_scores[doc_index]
            if score > 0:
                
                cleaned_doc_text = re.sub(r'<.*?>', '', self.corpus[doc_index])
                
                results.append({
                    'rank': rank + 1,
                    'title': self.titles[doc_index],
                    'score': score,
                   
                    'text_snippet': cleaned_doc_text[:100].strip() + "..."
                })
        
        print(f"\nQuery: '{query}'")
        print(f"Retrieval time: {query_speed:.6f} seconds.")
        
        return results



def run_system():
    global CORPUS, DOCUMENT_TITLES
    
    
    try:
        from nltk.stem.porter import PorterStemmer
    except ImportError:
        print("FATAL ERROR: The 'nltk' library is required for stemming.")
        print("Please install it: pip install nltk")
        print("You may also need to download the tokenizers: import nltk; nltk.download('punkt')")
        return

    print("Initializing Information Retrieval System...")
    
    
    CORPUS, DOCUMENT_TITLES = load_corpus_from_csv(CORPUS_FILE)
    
    if not CORPUS:
        print("\nFATAL: No documents loaded. Please check that 'Articles.csv' exists and has valid content.")
        return
        
    
    print(f"DEBUG: First document content (Title: {DOCUMENT_TITLES[0]}):")
   
    print(f"       Snippet: {CORPUS[0][:150].strip().replace('\n', ' ')}...")
    print("-" * 50)
   
    ir_system = TFIDF_IR_System(CORPUS, DOCUMENT_TITLES)
    
   
    ir_system.build_index()
    
    
    queries = [
        "Sindh Government", 
        "methods for text analysis",
        "evaluation of system performance" 
    ]
    
    print("\n--- Running Example Queries ---")
    for query in queries:
        top_k = 3
        results = ir_system.retrieve(query, k=top_k)
        
        if isinstance(results, str):
            print(f"Query: '{query}' -> {results}")
            continue

        if not results:
            print(f"Query: '{query}' -> No relevant documents found.")
            continue
            
        print(f"\nTop {len(results)} Results for Query: '{query}'")
        print("-" * 50)
        for result in results:
            clean_title = re.sub(r'<.*?>', '', result['title'])
           
            clean_snippet = result['text_snippet'] 
            
            print(f"Rank {result['rank']} (Score: {result['score']:.4f})")
            print(f"  Title: {clean_title}")
            print(f"  Snippet: {clean_snippet}")
        print("\n" + "=" * 50)

if __name__ == "__main__":
    
    try:
        run_system()
    except ModuleNotFoundError as e:
        print(f"\nFATAL ERROR: Required library not found. Please install it:")
        if 'sklearn' in str(e) or 'numpy' in str(e):
            print("Run: pip install scikit-learn numpy")
        elif 'nltk' in str(e):
            print("Run: pip install nltk")
            print("You may also need to download the tokenizers: import nltk; nltk.download('punkt')")
        else:
            print(f"Unknown module error: {e}")
    except Exception as e:
        if "empty vocabulary" in str(e):
            print(f"An unexpected error occurred: {e}")
            print("\nSUGGESTION: The documents might be too short or contain only non-alphabetic characters. This usually means the content_index in load_corpus_from_csv is pointing to a wrong column.")
        else:
            print(f"An unexpected error occurred: {e}")