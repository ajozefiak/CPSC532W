import numpy as np

def sample_topic_assignment(topic_assignment,
                            topic_counts,
                            doc_counts,
                            topic_N,
                            doc_N,
                            alpha,
                            gamma,
                            words,
                            document_assignment):
    """
    Sample the topic assignment for each word in the corpus, one at a time.

    Args:
        topic_assignment: size n array of topic assignments
        topic_counts: n_topics x alphabet_size array of counts per topic of unique words
        doc_counts: n_docs x n_topics array of counts per document of unique topics

        topic_N: array of size n_topics count of total words assigned to each topic
        doc_N: array of size n_docs count of total words in each document, minus 1

        alpha: prior dirichlet parameter on document specific distributions over topics
        gamma: prior dirichlet parameter on topic specific distribuitons over words.

        words: size n array of wors
        document_assignment: size n array of assignments of words to documents
    Returns:
        topic_assignment: updated topic_assignment array
        topic_counts: updated topic counts array
        doc_counts: updated doc_counts array
        topic_N: updated count of words assigned to each topic
    """

    # Number of topics
    num_topics = topic_counts.shape[0]

    # Number of words
    num_words = len(words)

    #alphabet_size
    alphabet_size = topic_counts.shape[1]

    # sample topic assignment for each of the n words in turn
    for n in range(num_words):

        # Document of nth word
        doc = document_assignment[n]
        # Topic assignment of nth word
        topic = topic_assignment[n]
        # alphabet-word of nth word
        word = words[n]

        prob_sum = 0.0
        prob_dist = np.zeros(num_topics)

        # sample a new topic
        for k in range(num_topics):
            # compute probability of p(z_i,d <= k|...)
            if k == topic:
               t_1 = (alpha + doc_counts[doc,k]-1)/(num_topics*alpha + np.sum(doc_counts[doc,:])-1)
               t_2 = (gamma + topic_counts[k,word]-1) / (alphabet_size*gamma + np.sum(topic_counts[k,:])-1)
            else:
               t_1 = (alpha + doc_counts[doc,k])/(num_topics*alpha + np.sum(doc_counts[doc,:])-1)
               t_2 = (gamma + topic_counts[k,word]) / (alphabet_size*gamma + np.sum(topic_counts[k,:])-1)

            prob_sum += (t_1*t_2)
            prob_dist[k] = t_1*t_2

        prob_dist = np.cumsum(prob_dist / np.sum(prob_dist))
        prob = np.random.uniform(0,1)

        for k in range(num_topics):

            # we sample z_id to be topic k
            if prob <= prob_dist[k]:

                # Update topic_assignment
                topic_assignment[n] = k

                topic_counts[topic,word] -= 1
                topic_counts[k,word] += 1

                doc_counts[doc,topic] -= 1
                doc_counts[doc,k] += 1

                topic_N[topic] -= 1
                topic_N[k] += 1

                break

    return (topic_assignment, topic_counts, doc_counts, topic_N)
