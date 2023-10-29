from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import CountVectorizer
from gensim.corpora import Dictionary
import numpy as np

def cluster_data_nmf(df):
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
        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init='auto')
        cluster_labels = kmeans.fit_predict(tfidf_matrix)
        score = silhouette_score(tfidf_matrix, cluster_labels)
        silhouette_scores.append(score)
        if score > best_score:
            best_score = score
            best_num_clusters = n_clusters

    # Perform clustering with the best number of clusters
    kmeans = KMeans(n_clusters=best_num_clusters, random_state=0, n_init='auto')
    df['cluster_id'] = kmeans.fit_predict(tfidf_matrix)
    # Extract group names and descriptions using NMF (Non-Negative Matrix Factorization)
    num_topics = best_num_clusters

    # Create a gensim dictionary and corpus for NMF
    corpus = [text.split() for text in df["text"]]
    dictionary = Dictionary(corpus)
    corpus = [dictionary.doc2bow(text) for text in corpus]

    # Reapply NMF with the optimal number of topics
    nmf = NMF(n_components=num_topics, random_state=0)
    nmf_topic_matrix = nmf.fit_transform(tfidf_matrix)

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

        group_name = "Cluster " + str(i)  # Use NMF topic index as group name
        group_description = f"Cluster {i} - {topic_keywords}"

        group_names_dict[i] = group_name
        group_descriptions_dict[i] = group_description
    
    group_descriptions = [group_descriptions_dict[label] for label in df["cluster_id"]]
    group_names = [group_names_dict[label] for label in df["cluster_id"]]

    # Assign cluster names and unique group descriptions to the data
    df["group_name"] = group_names
    df["group_description"] = group_descriptions
    df.drop(columns = ['text', 'cluster_id'], inplace = True)
    return df