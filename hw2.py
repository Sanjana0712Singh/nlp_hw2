# ----- PART 1 ----- #
# Bigram Language Model to detect which sentence is randomly generated

import math, re

# Global data structures used by both training and testing
bigram_counts = {} # Counts of (word1, word2)
unigram_counts = {} # Counts of single words
vocab = set() # Set of all unique words in the training corpus

def calcNGrams_train(trainFile):
    """
    Trains a bigram language model from a text file. 
    Populates unigram_counts, bigram_counts, and vocab.
    """
    
    global bigram_counts
    global unigram_counts
    global vocab
    
    # Reset globals in case training is called multiple times
    bigram_counts = {}
    unigram_counts = {}
    vocab = set()
    
    try:
        with open(trainFile, "r", encoding="utf-8") as f:
            for line in f:
                # Tokenize: keep only letters and apostrophes, convert to lowercase
                words = re.findall(r"[a-zA-Z']+", line.lower())
                
                # Skip empty lines if there are any
                if not words: 
                    continue
                
                # Count unigrams (single words)
                for word in words:
                    vocab.add(word)
                    unigram_counts[word] = unigram_counts.get(word, 0) + 1
                
                # Count bigrams (pairs of consecutive words)
                for i in range(len(words) - 1):
                    bigram = (words[i], words[i+1])
                    bigram_counts[bigram] = bigram_counts.get(bigram, 0) + 1
    except FileNotFoundError:
        pass

def calcNGrams_test(sentences):
    """
    Given a list of sentences, return the index of the sentence
    that is most likely randomly generated.
    """

    V = len(vocab)  # Vocabulary size (needed for Laplace smoothing)
    sentence_scores = []
    
    for sentence in sentences:
        # Tokenize the sentence the same way as during training
        words = re.findall(r"[a-zA-Z']+", sentence.lower())
 
        log_prob = 0
        
        # Compute probability using bigram model
        for i in range(len(words) - 1):
            bigram = (words[i], words[i+1])
            
            # Get counts (0 if unseen)
            bigram_count = bigram_counts.get(bigram, 0)
            unigram_count = unigram_counts.get(words[i], 0)
            
            # Laplace smoothing:
            # P(w2 | w1) = (count(w1,w2) + 1) / (count(w1) + V)
            prob = (bigram_count + 1) / (unigram_count + V)
            
            # Use log probabilities to avoid underflow
            log_prob += math.log(prob)
        
        # Normalize by sentence length so long sentences aren't penalized
        avg_log_prob = log_prob / (len(words) - 1)
        sentence_scores.append(avg_log_prob)
    
    # The random sentence will have the lowest probability
    random_index = sentence_scores.index(min(sentence_scores))
    
    return random_index


# ----- PART 2 ----- #
# Sentiment classification using Naive Bayes

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import json

# Global variables storing trained model
vectorizer = None
model = None

def calcSentiment_train(trainFile):
    """
    Trains a Naïve Bayes sentiment classifier.
    """

    global vectorizer
    global model

    reviews = [] # Text reviews
    labels = [] # Corresponding sentiment labels

    try: 
        # Read training data
        with open(trainFile, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                reviews.append(data["review"])
                labels.append(data["sentiment"])
    except FileNotFoundError:
        pass

    # Convert text into numerical features (Bag-of-Words)
    vectorizer = CountVectorizer(
        stop_words="english", # Removes common words like "the", "is"
        ngram_range=(1,2)  # Uses both single words and word pairs
    )

    # Learn vocabulary and transform training reviews into feature matrix
    X = vectorizer.fit_transform(reviews)

    # Train a Multinomial Naïve Bayes classifier
    model = MultinomialNB(alpha=0.1) # Alpha applies smoothing to avoid zero probabilities
    model.fit(X, labels)

def calcSentiment_test(review):
    """
    Predict sentiment for a single review.
    """

    global vectorizer
    global model

    # Convert review into feature vector using the trained vectorizer
    X = vectorizer.transform([review])
    
    # Predict sentiment
    prediction = model.predict(X)

    # Return Boolean result
    return bool(prediction[0])