import numpy as np

def joint_log_lik(doc_counts, topic_counts, alpha, gamma):
    """
    Calculate the joint log likelihood of the model

    Args:
        doc_counts: n_docs x n_topics array of counts per document of unique topics
        topic_counts: n_topics x alphabet_size array of counts per topic of unique words
        alpha: prior dirichlet parameter on document specific distributions over topics
        gamma: prior dirichlet parameter on topic specific distribuitons over words.
    Returns:
        jll: the joint log likelihood of the model
    """
    num_docs = doc_counts.shape[0]
    num_topics = topic_counts.shape[0]
    alphabet_size = topic_counts.shape[1]

    jll = 0.0
    for i in range(num_docs):
        jll -=  num_topics * np.log(num_topics*alpha + np.sum(doc_counts[i,:]))
        for j in range(num_topics):
            jll += np.log((alpha + doc_counts[i,j]))

    for i in range(num_topics):
        jll -= alphabet_size * np.log(alphabet_size*gamma + np.sum(topic_counts[i,:]))
        for j in range(alphabet_size):
            jll += np.log((gamma + topic_counts[i,j]))

    return jll
