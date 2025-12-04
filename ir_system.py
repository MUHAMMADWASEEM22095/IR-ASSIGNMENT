import re
import time
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


CORPUS_DIR = 'corpus_data'

# The following lists will be populated by the file loading function.
CORPUS = []
DOCUMENT_TITLES = []

def load_corpus_from_disk(directory):
    """
    Loads all text files from a specified directory into the CORPUS and DOCUMENT_TITLES lists.
    """
    print(f"Loading documents from directory: {directory}")
    local_corpus = []
    local_titles = []
    
    # Check if the directory exists
    if not os.path.isdir(directory):
        print(f"Error: Directory '{directory}' not found. Please create it and place your text files inside.")
        return [], []

    # Iterate through all files in the directory
    for filename in os.listdir(directory):
        # We only want to process text files (assuming .txt extension)
        if filename.endswith('.txt'):
            filepath = os.path.join(directory, filename)
            try:
                # Open and read the entire content of the file
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                    # Add the file content to the corpus
                    local_corpus.append(content)
                    
                    # Use the filename (without extension) as the document title
                    title = filename.replace('.txt', '')
                    local_titles.append(title)
                
            except Exception as e:
                print(f"Could not read file {filename}. Error: {e}")
                
    print(f"Successfully loaded {len(local_corpus)} documents.")
    return local_corpus, local_titles
class TFIDF_IR_System:
    def __init__(self, corpus, titles):
        
        self.corpus = corpus
        self.titles = titles
        self.vectorizer = None
        self.doc_vectors = None

    def preprocess(self, text):
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)
        return text

    def convert_to_index(self):
        print("--- Indexing Phase Started ---")
        processed_corpus = [self.preprocess(doc) for doc in self.corpus]
        
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            max_df=0.85, 
            norm='l2' 
        )
     
        start_time = time.time()
        self.doc_vectors = self.vectorizer.fit_transform(processed_corpus)
        indexing_time = time.time() - start_time
        
        print(f"Total documents indexed: {len(self.corpus)}")
        print(f"Vocabulary size: {len(self.vectorizer.get_feature_names_out())}")
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

        start_time = time.time()
        similarity_scores = cosine_similarity(query_vector, self.doc_vectors).flatten()
        query_speed = time.time() - start_time
        ranked_indices = np.argsort(similarity_scores)[::-1]
        
        results = []
        for rank, doc_index in enumerate(ranked_indices[:k]):
            score = similarity_scores[doc_index]
            if score > 0:
                results.append({
                    'rank': rank + 1,
                    'title': self.titles[doc_index],
                    'score': score,
                    'text_snippet': self.corpus[doc_index][:100] + "..."
                })
        
        print(f"\nQuery: '{query}'")
        print(f"Retrieval time: {query_speed:.6f} seconds.")
        
        return results


def run_system():
    print("Initializing Information Retrieval System...")
    
    ir_system = TFIDF_IR_System(CORPUS, DOCUMENT_TITLES)
    
    ir_system.convert_to_index()
    
    queries = [
        "documents relevant to VSM and space",
        "methods for analyzing text patterns",
        "ranking criteria precision"
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
            print(f"Rank {result['rank']} (Score: {result['score']:.4f})")
            print(f"  Title: {result['title']}")
            print(f"  Snippet: {result['text_snippet']}")
        print("\n" + "=" * 50)

if __name__ == "__main__":

    try:
        run_system()
    except ModuleNotFoundError as e:
        print(f"\nFATAL ERROR: Required library not found. Please install it:")
        if 'sklearn' in str(e):
            print("Run: pip install scikit-learn numpy")
        else:
            print(f"Unknown module error: {e}")