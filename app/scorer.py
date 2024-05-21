from nltk.stem.porter import PorterStemmer
from nltk.tokenize import TreebankWordTokenizer
import nltk
import string
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter


class Scorer:
    def __init__(self, docs):
        self.docs = docs
        self.vectorizer = TfidfVectorizer(...)
        self.doc_tfidf = self.vectorizer.fit_transform(docs)
        self.feedback = {}

    def score(self, query):
        feature_vectors = [self._feature_tfidf(
            query), self._feature_positive_feedback(query)]
        return np.sum([feature * weight for feature, weight in zip(feature_vectors, self.feature_weights)], axis=0)

    def learn_feedback(self, feedback_dict):
        self.feedback = feedback_dict

    def _feature_tfidf(self, query):
        query_vector = self.vectorizer.transform([query])
        similarity = cosine_similarity(query_vector, self.doc_tfidf)
        return similarity.ravel()


class Scorer:
    def __init__(self, docs):
        self.docs = docs
        self.vectorizer = TfidfVectorizer(
            tokenizer=self.tokenize_and_stem, stop_words='english')
        self.doc_tfidf = self.vectorizer.fit_transform(docs)
        self.features = [self._feature_tfidf, self._feature_positive_feedback]
        self.feature_weights = [1., 2.]
        self.feedback = {}

    def score(self, query):
        feature_vectors = [feature(query) for feature in self.features]
        feature_vectors_weighted = [
            feature * weight for feature, weight in zip(feature_vectors, self.feature_weights)]
        return np.sum(feature_vectors_weighted, axis=0)

    def learn_feedback(self, feedback_dict):
        self.feedback = feedback_dict

    def tokenize_and_stem(self, text):
        tokens = self.tokenizer.tokenize(text.lower().translate(
            str.maketrans('', '', string.punctuation)))
        return [self.stemmer.stem(token) for token in tokens]

    def _feature_tfidf(self, query):
        query_vector = self.vectorizer.transform([query])
        similarity = cosine_similarity(query_vector, self.doc_tfidf)
        return similarity.ravel()

    def _feature_positive_feedback(self, query):
        if not self.feedback:
            return np.zeros(len(self.docs))
        feedback_queries = list(self.feedback.keys())
        similarity = cosine_similarity(self.vectorizer.transform(
            [query]), self.vectorizer.transform(feedback_queries))
        nn_similarity = np.max(similarity)
        nn_idx = np.argmax(similarity)
        pos_feedback_doc_idx = [
            idx for idx, feedback_value in self.feedback[feedback_queries[nn_idx]] if feedback_value == 1.]
        counts = Counter(pos_feedback_doc_idx)
        pos_feedback_proportions = {
            doc_idx: count / sum(counts.values()) for doc_idx, count in counts.items()}
        feature_values = {doc_idx: nn_similarity * count / sum(
            counts.values()) for doc_idx, count in Counter(pos_feedback_doc_idx).items()}
        return np.array([feature_values.get(doc_idx, 0.) for doc_idx, _ in enumerate(self.docs)])

    @property
    def tokenizer(self):
        return TreebankWordTokenizer()

    @property
    def stemmer(self):
        return PorterStemmer()
