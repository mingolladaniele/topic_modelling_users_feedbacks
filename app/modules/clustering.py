from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import CountVectorizer
from gensim.corpora import Dictionary
import numpy as np
from sentence_transformers import SentenceTransformer
import pandas as pd
import umap
import hdbscan
import os


def c_tf_idf(documents, m, ngram_range=(1, 1)):
    """
    Compute the c-TF-IDF (normalized term frequency-inverse document frequency) matrix.

    Args:
        documents: List of text documents.
        m: Total number of documents.
        ngram_range: The n-gram range for the CountVectorizer.

    Returns:
        The computed c-TF-IDF matrix and the CountVectorizer object.
    """
    count = CountVectorizer(ngram_range=ngram_range, stop_words="english").fit(
        documents
    )
    t = count.transform(documents).toarray()
    w = t.sum(axis=1)
    tf = np.divide(t.T, w)
    sum_t = t.sum(axis=0)
    idf = np.log(np.divide(m, sum_t)).reshape(-1, 1)
    tf_idf = np.multiply(tf, idf)
    return tf_idf, count


def extract_top_n_words_per_topic(tf_idf, count, docs_per_topic, n=20):
    """
    Extract the top n words per topic from the c-TF-IDF matrix.

    Args:
        tf_idf: The c-TF-IDF matrix.
        count: The CountVectorizer object.
        docs_per_topic: DataFrame with documents grouped by topics.
        n: Number of top words to extract per topic.

    Returns:
        A dictionary containing the top n words for each topic.
    """
    words = count.get_feature_names_out()
    labels = list(docs_per_topic.Topic)
    tf_idf_transposed = tf_idf.T
    indices = tf_idf_transposed.argsort()[:, -n:]
    top_n_words = {
        label: [(words[j], tf_idf_transposed[i][j]) for j in indices[i]][::-1]
        for i, label in enumerate(labels)
    }
    return top_n_words


def extract_topic_sizes(df):
    """
    Extract the sizes of each topic in the DataFrame.

    Args:
        df: DataFrame with clustered data.

    Returns:
        DataFrame with the size of each topic.
    """
    topic_sizes = (
        df.groupby(["Topic"])
        .text.count()
        .reset_index()
        .rename({"Topic": "Topic", "text": "Size"}, axis="columns")
        .sort_values("Size", ascending=False)
    )
    return topic_sizes


def cluster_data_bert(df):
    """
    Cluster data using the BERT sentence embeddings.

    Args:
        df: DataFrame containing text data.

    Returns:
        DataFrame with clustered data.
    """
    model_dir = "./sentence_transformer_model"
    if not os.path.exists(model_dir):
        # Download the model if it doesn't exist
        model = SentenceTransformer("distilbert-base-nli-mean-tokens")
        model.save(model_dir)
    else:
        # Load the model from the existing directory
        model = SentenceTransformer(model_dir)

    embeddings = model.encode(df["text"], show_progress_bar=False)

    umap_embeddings = umap.UMAP(
        n_neighbors=15, n_components=10, metric="cosine"
    ).fit_transform(embeddings)

    cluster = hdbscan.HDBSCAN(
        min_cluster_size=4, metric="euclidean", cluster_selection_method="eom"
    ).fit(umap_embeddings)
    docs_df = pd.DataFrame(df, columns=["text"])

    docs_df["Topic"] = cluster.labels_
    docs_df["Doc_ID"] = range(len(docs_df))
    docs_per_topic = docs_df.groupby(["Topic"], as_index=False).agg({"text": " ".join})
    tf_idf, count = c_tf_idf(docs_per_topic.text.values, m=len(df))
    top_n_words = extract_top_n_words_per_topic(tf_idf, count, docs_per_topic, n=5)
    # we can use topic_sizes to view how frequent certain topics are
    # topic_sizes = extract_topic_sizes(docs_df)

    df["group_name"] = ["cluster " + str(cluster_id) for cluster_id in cluster.labels_]
    df["group_description"] = [
        ",".join([word for word, _ in top_n_words[cluster_id]]) for cluster_id in cluster.labels_
    ]
    df.drop(columns=["text"], inplace=True)
    return df


def cluster_data_nmf(df):
    """
    Cluster data using Non-Negative Matrix Factorization (NMF).

    Args:
        df: DataFrame containing text data.

    Returns:
        DataFrame with clustered data.
    """
    # TF-IDF Vectorization with improved parameters
    tfidf_vectorizer = TfidfVectorizer(
        max_df=0.7, max_features=50, lowercase=True, stop_words="english"
    )
    tfidf_matrix = tfidf_vectorizer.fit_transform(df["text"])

    # Initialize variables to store silhouette scores
    silhouette_scores = []

    # Determine the best number of clusters using silhouette score
    max_clusters = 15  # You can adjust this based on your data
    best_score = -1
    best_num_clusters = 0
    for n_clusters in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto")
        cluster_labels = kmeans.fit_predict(tfidf_matrix)
        score = silhouette_score(tfidf_matrix, cluster_labels)
        silhouette_scores.append(score)
        if score > best_score:
            best_score = score
            best_num_clusters = n_clusters

    # Perform clustering with the best number of clusters
    kmeans = KMeans(n_clusters=best_num_clusters, random_state=0, n_init="auto")
    df["cluster_id"] = kmeans.fit_predict(tfidf_matrix)
    # Extract group names and descriptions using NMF (Non-Negative Matrix Factorization)
    num_topics = best_num_clusters

    # Create a gensim dictionary and corpus for NMF
    corpus = [text.split() for text in df["text"]]
    dictionary = Dictionary(corpus)
    corpus = [dictionary.doc2bow(text) for text in corpus]

    # Apply NMF with the optimal number of topics
    nmf = NMF(n_components=num_topics, random_state=0)

    # Find keywords for each topic
    vectorizer = CountVectorizer(analyzer="word", stop_words="english")
    keywords = vectorizer.fit(df["text"]).get_feature_names_out()
    group_names_dict = {}
    group_descriptions_dict = {}

    for i in range(num_topics):
        # Extract top keywords for the group description
        topic_keywords = " ".join(
            keywords[idx] for idx in np.argsort(nmf.components_[i])[-10:][::-1]
        )
        
        # Use NMF topic index as group name
        group_name = "cluster " + str(i)
        group_description = f"cluster {i} - {topic_keywords}"

        group_names_dict[i] = group_name
        group_descriptions_dict[i] = group_description

    group_descriptions = [group_descriptions_dict[label] for label in df["cluster_id"]]
    group_names = [group_names_dict[label] for label in df["cluster_id"]]

    # Assign cluster names and unique group descriptions to the data
    df["group_name"] = group_names
    df["group_description"] = group_descriptions
    df.drop(columns=["text", "cluster_id"], inplace=True)
    return df
