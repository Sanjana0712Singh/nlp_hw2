README: NLP Homework 2: N-grams and Naive Bayes
_____________________
| Group Members:    |
| Skyla Fitzgerald  |
| Sanjana Singh     |

___________________________________________________________________________________
AI Disclosure
___________________________________________________________________________________

During the completion of this project, the following uses of AI tools 
(Gemini/ChatGPT) were made in compliance with the course policy:

Theory portion: AI tools were used to review and refine the wording of written explanations to ensure clarity and correctness.

Program Implementation: AI was used to help understand certain library components used in the implementation, specifically CountVectorizer and MultinomialNB from the sklearn library.

Program Testing: AI tools were used to generate and test additional example sentences and paragraphs similar to the provided examples in order to verify that the program behaved as expected.

___________________________________________________________________________________
High-Level Design Decisions
___________________________________________________________________________________

1. Normalization by Sentence Length
	
	When computing the probability of a sentence using the bigram language model, we summed the log 	probabilities of each bigram. However, longer sentences naturally contain more bigrams, which would cause them to accumulate more negative log probabilities and appear less likely even if they are grammatically correct.

	To avoid penalizing longer sentences unfairly, we normalize the total log probability by the number of bigrams in the sentence. This produces an average log probability per bigram, allowing sentences of different lengths to be compared more fairly.

2. Tokenization and Laplace Smoothing
	
	During training, text is tokenized using a regular expression that keeps only alphabetic characters and apostrophes. All tokens are converted to lowercase to ensure that words such as “The” and “the” are treated as the same token.

	When computing bigram probabilities, we apply Laplace smoothing to prevent zero probabilities for unseen bigrams, which would otherwise make the entire sentence probability zero.

3. Use of Unigrams and Bigrams for Sentiment Features

	The vectorizer is configured with: ngram_range=(1,2). This means the classifier uses both unigrams and bigrams (two-word phrases). Including bigrams allows the model to capture short contextual phrases such as “not good” or “very bad”, which can be important indicators of sentiment.

4. Multinomial Naïve Bayes

	We used Multinomial Naïve Bayes (MultinomialNB) because it is a standard and effective model for text classification problems using bag-of-words features. This model works well when features represent word counts, which is exactly the representation produced by CountVectorizer.
	
	The classifier is initialized with: MultinomialNB(alpha=0.1). The alpha parameter applies additive smoothing, which prevents the model from assigning zero probability to words that were not seen in a particular class during training. Using a small value such as 0.1 provides smoothing while still allowing the model to strongly weight informative features.