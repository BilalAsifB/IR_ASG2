# Vector Space Model (VSM) Search Engine

# Imports
import glob
import os
import json
import math
from collections import defaultdict, Counter
import spacy
from spacy.language import Language
import tkinter as tk
from tkinter import scrolledtext


def read_all_docs(folder_path):
    """
    Reads all .txt files from the specified folder path.
    """
    folder_path += "" if folder_path[-1] == '/' else '/'
    return glob.glob(folder_path + '*.txt')

def get_stop_wrds():
    """
    Reads a custom stopword list from a file and returns it as a set.
    """
    with open('./resources/Stopword-list.txt', 'r') as f:
        return {wrd.lower().strip() for wrd in f.readlines()}

def get_text(path):
    """
    Reads the content of a text file and returns it as a lowercase string.
    """
    with open(path, 'r', encoding="utf-8", errors="ignore") as f:
        return f.read().lower()

@Language.component("stop_words_remover")
def stop_words_remover(doc):
    """
    Custom pipeline component to remove stop words and punctuation from the document.
    """
    return spacy.tokens.Doc(doc.vocab, words=[token.text for token in doc if not token.is_stop and not token.is_punct])

def prepare_pipeline():
    """
    Prepares the spaCy pipeline with a custom stop word remover component.
    """
    custom_stop_words = get_stop_wrds()
    nlp = spacy.load("en_core_web_sm")
    nlp.Defaults.stop_words.update(custom_stop_words)
    nlp.add_pipe("stop_words_remover", first=True)
    return nlp

def compute_tf_idf(nlp, save=True):
    """
    Computes the TF-IDF vectors for documents in the specified folder and saves them to files.
    """
    path = "./resources/Abstracts/Abstracts"
    txt_files = read_all_docs(path)

    global doc_vectors, idf_scores, documents, positional_index
    documents = set()
    term_doc_freq = defaultdict(int)
    tf_per_doc = {}
    doc_vectors = {}
    positional_index = defaultdict(lambda: defaultdict(list))
    total_docs = len(txt_files)

    for file in txt_files:
        doc_name = os.path.basename(file).replace('.txt', '')
        documents.add(doc_name)
        text = get_text(file)
        doc = nlp(text)

        tokens = [token.lemma_ for token in doc if token.text.strip()]
        tf_per_doc[doc_name] = Counter(tokens)

        for pos, token in enumerate(doc):
            if token.text.strip():
                lemma = token.lemma_
                positional_index[lemma][doc_name].append(pos)

        for term in set(tokens):
            term_doc_freq[term] += 1

    idf_scores = {
        term: math.log(total_docs / df) for term, df in term_doc_freq.items()
    }

    for doc_name, term_freqs in tf_per_doc.items():
        vector = {}
        for term, freq in term_freqs.items():
            idf = idf_scores.get(term, 0)
            vector[term] = freq * idf
        doc_vectors[doc_name] = vector

    if save:
        with open("./data/tfidf_index.json", "w") as f:
            json.dump(doc_vectors, f)
        with open("./data/idf_scores.json", "w") as f:
            json.dump(idf_scores, f)
        with open("./data/positional_index.json", "w") as f:
            json.dump({k: dict(v) for k, v in positional_index.items()}, f)

def load_vectors():
    """
    Loads the TF-IDF vectors, IDF scores, and positional index from JSON files.
    """
    global doc_vectors, idf_scores, documents, positional_index
    with open("./data/tfidf_index.json", "r") as f:
        doc_vectors = json.load(f)
    with open("./data/idf_scores.json", "r") as f:
        idf_scores = json.load(f)
    with open("./data/positional_index.json", "r") as f:
        positional_index = json.load(f)
    documents = set(doc_vectors.keys())

def cosine_similarity(vec1, vec2):
    """
    Computes the cosine similarity between two vectors.
    """
    dot = sum(vec1.get(term, 0) * vec2.get(term, 0) for term in set(vec1) | set(vec2))
    norm1 = math.sqrt(sum(v ** 2 for v in vec1.values()))
    norm2 = math.sqrt(sum(v ** 2 for v in vec2.values()))
    return dot / (norm1 * norm2) if norm1 and norm2 else 0.0

def contains_phrase(doc_name, phrase_tokens):
    """
    Checks if a document contains a specific phrase using the positional index.
    """
    if any(token not in positional_index or doc_name not in positional_index[token] for token in phrase_tokens):
        return False

    positions_lists = [positional_index[token][doc_name] for token in phrase_tokens]
    for p in positions_lists[0]:
        if all((p + i) in positions_lists[i] for i in range(len(phrase_tokens))):
            return True
    return False

def process_query(nlp, query, alpha=0.05):
    """
    Processes the query, computes the TF-IDF vector, and retrieves relevant documents.
    """
    is_phrase = query.startswith('"') and query.endswith('"')
    query = query.strip('"')

    query_doc = nlp(query.lower())
    query_terms = [token.lemma_ for token in query_doc if token.text.strip()]
    query_tf = Counter(query_terms)

    query_vector = {
        term: freq * idf_scores.get(term, 0)
        for term, freq in query_tf.items()
    }

    doc_scores = {}
    doc_vectors_float = {doc: {k: float(v) for k, v in vec.items()} for doc, vec in doc_vectors.items()}

    for doc, vector in doc_vectors_float.items():
        sim = cosine_similarity(query_vector, vector)
        if sim >= alpha:
            if is_phrase:
                if contains_phrase(doc, query_terms):
                    doc_scores[doc] = sim
            else:
                doc_scores[doc] = sim

    return sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

def search_query():
    """
    Handles the search query from the GUI, processes it, and displays results.
    """
    query = query_entry.get()
    results = process_query(nlp, query)

    result_text.delete(1.0, tk.END)
    result_text.insert(tk.END, f"Results for query: {query}\n\n", "header")

    if results:
        for doc, score in results:
            result_text.insert(tk.END, f"{doc} (Score: {score:.4f})\n", "result")
    else:
        result_text.insert(tk.END, "No results found.\n", "result")

def create_gui():
    """
    Creates the GUI for the search engine using Tkinter.
    """
    global query_entry, result_text

    root = tk.Tk()
    root.title("Bilal's VSM Search Engine")
    root.geometry("500x500")
    root.configure(bg="#004c99")

    tk.Label(root, text="Enter query:", fg="White", bg="#004c99", font=("Arial", 30, "bold")).pack(pady=3)
    query_entry = tk.Entry(root, width=80, font=("Arial", 30))
    query_entry.pack(pady=3)

    search_button = tk.Button(root, text="Search", command=search_query, fg="White", bg="#004c99", font=("Arial", 30, "bold"))
    search_button.pack(pady=3)

    result_text = scrolledtext.ScrolledText(root, width=70, height=25, font=("Arial", 15), bg="white")
    result_text.pack(pady=5)

    root.mainloop()

if __name__ == "__main__":
    nlp = prepare_pipeline()
    try:
        load_vectors()
    except FileNotFoundError:
        compute_tf_idf(nlp)
    create_gui()
