# Natural language normalization
import re
import math




######################################################################
#                                                                    #
#                             Files API                              #
#                                                                    #
######################################################################

# String text letters and numbers only.
def StringText(text:str,lower:bool, alphanum:bool)->str:
#	try:
	norm_text = text
	if (lower):
		norm_text = norm_text.lower()
	if (alphanum):
		norm_text = re.sub('[^a-zA-Z]+', ' ', norm_text)
		#norm_text = re.sub('[^0-9a-zA-Z]+', ' ', norm_text)
	norm_text = " ".join(norm_text.split())		# remove trailing,ecseeding and multiple white spaces.
	return norm_text

# String text letters and numbers only.
def StringLetters(text:str)->str:
	norm_text = re.sub('[^a-zA-Z]+', ' ', text)	# substitute everything not letters with space.
	norm_text = " ".join(norm_text.split())		# remove trailing,ecseeding and multiple white spaces.
	return norm_text


# Normalize the text.
# returning the text containint alphanum text.
# and remove hihly frequent words (stop words).
def DocsNorm(docs_dic:dict, lower:bool, alphanum:bool)->dict:
	docs_norm = {}
	for key, document in docs_dic.items():
		norm_text = StringText(document, lower, alphanum)
		docs_norm[key] = norm_text
	return docs_norm

#Convert the list of document terms in to a unified list of terms.
def DocsWordset(docs_dic:dict):
	wordSet = {}
	for key, doc in docs_dic.items():
		for word in doc.split():
			wordSet[word] = word
	dictionary = set(wordSet.keys())
	return dictionary

def DocsTermFreq(docs_dic:list, wordSet):
	docTermFreq = {}
	for doc in docs_dic:
		docTermFreq[doc] = dict.fromkeys(wordSet, 0)   # Create a list of document terms with count 0.

	for key,doc in docs_dic.items():
		for word in doc.split():									# Loop tokens in document.
			docTermFreq[key][word] += 1				# Increase the number of occurrences for each term.
	return docTermFreq

# Dictionary word frequency.
# The frequency of vocabulary words in the corpus.
def VocabWordfrq(docs_dic:dict, vocab_terms:list):
	vocab_freq = dict.fromkeys(vocab_terms, 0)	#Vector of dictionary items.
	for key, doc in docs_dic.items():
		for word, val in doc.items():
			if val > 0:
				vocab_freq[word] += val								# count occurances if each term on each document.
	return vocab_freq


# Compute the inverse document frequence for each document term.
def VocabInvDocfrq(N:int, vocab_freq:dict):
	idfDict = dict.fromkeys(vocab_freq.keys(), 0)

	for word, val in vocab_freq.items():
		if(val > 0):
			idfDict[word] = math.log10(N / float(val))
	return idfDict

#Smoothed inverse document frequency.
def VocabInvDocfrqs(N:int, vocab_freq:dict):
	idfDict = dict.fromkeys(vocab_freq.keys(), 0)
	for word, val in vocab_freq.items():
		idfDict[word] = math.log10(N / float(1+val)) + 1
	return idfDict

#Probobalistic inverse document frequency.
def VocabInvDocfrqp(N:int, vocab_freq:dict):
	idfDict = dict.fromkeys(vocab_freq.keys(), 0)
	for word, val in vocab_freq.items():
		if(N - val):
			idfDict[word] = math.log10( (N - val) / float(val))
	return idfDict

import numpy
def QueryCosSim(query_terms:list, doc_terms:list):
	dot = numpy.dot(query_terms, doc_terms)

	norma = numpy.linalg.norm(query_terms)
	normb = numpy.linalg.norm(doc_terms)

	cos_sim = dot / (norma * normb)

	return cos_sim



# Create a list of terms for each of the documents.
#def DocsTerms(docs_text:dict):
#	docs_terms:dict={}
#	for doc in docs_text:
#		docs_terms[doc] = Cast.StringTerms(docs_text[doc]);
#	return(docs_terms);

# # Text Cleaning
# - remove the **mentions**, as we want to make the model generalisable to tweets.
# - remove the **hash tag sign** (#) but not the actual tag as this may contain information
# - set all words to **lowercase**
# - remove all **punctuations**, including the question and exclamation marks
# - remove the **urls** as they do not contain useful information.
# - make sure the converted **emojis** are kept as one word.
# - remove **digits**
# - remove **stopwords**
# - apply the **PorterStemmer** to keep the stem of the words.
import os
import sys
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.base import BaseEstimator
from sklearn.base import  TransformerMixin

sys.path.append(os.path.join(os.getcwd(),'../'))





class TweetsClean(BaseEstimator, TransformerMixin):

	def remove_mentions(self, input_text):
		return re.sub(r'@\w+', '', input_text)

	def remove_urls(self, input_text):
		return re.sub(r'http.?://[^\s]+[\s]?', '', input_text)

	def emoji_oneword(self, input_text):
		# By compressing the underscore, the emoji is kept as one word
		return input_text.replace('_', '')

	def remove_punctuation(self, input_text):
		# Make translation table
		punct = string.punctuation
		trantab = str.maketrans(punct, len(punct) * ' ')  # Every punctuation symbol will be replaced by a space
		return input_text.translate(trantab)

	def remove_digits(self, input_text):
		return re.sub('\d+', '', input_text)

	def to_lower(self, input_text):
		return input_text.lower()

	def remove_stopwords(self, input_text):
		stopwords_list = stopwords.words('english')
		# Some words which might indicate a certain sentiment are kept via a whitelist
		whitelist = ["n't", "not", "no"]
		words = input_text.split()
		clean_words = [word for word in words if (word not in stopwords_list or word in whitelist) and len(word) > 1]
		return " ".join(clean_words)

	def stemming(self, input_text):
		porter = PorterStemmer()
		words = input_text.split()
		stemmed_words = [porter.stem(word) for word in words]
		return " ".join(stemmed_words)

	def fit(self, X, y=None, **fit_params):
		return self

	def transform(self, X, **transform_params):
		clean_X = X.apply(self.remove_mentions).apply(self.remove_urls).apply(self.emoji_oneword).apply(
			self.remove_punctuation).apply(self.remove_digits).apply(self.to_lower).apply(self.remove_stopwords).apply(
			self.stemming)
		return clean_X


# First we combine the TweetsCounts statistics with the CleanText variable.
# It suffices to run them only once.
def TextCleanSet(df_eda, text_clean, text_name:str):
	data_frame_model = df_eda
	data_frame_model[text_name] = text_clean
	data_frame_model.columns.tolist()
	return(data_frame_model)





# **NOTE: **One side-effect of text cleaning is that some rows do not have any words left in their text.
# for the Word2Vec algorithm this causes an error.
#  missing values will impute with a placeholder text '[no_text]'.
#def CleanTextRun(data_frame, text_col, empty_text,out_path:path, show:bool):
def CleanTextRun(data_frame, text_col, empty_text):
	text_clean = TweetsClean().fit_transform(data_frame[text_col])
	text_clean.sample(5)

	empty_clean = text_clean == ''
	print('{} records have no words left after text cleaning'.format(text_clean[empty_clean].count()))
	text_clean.loc[empty_clean] = empty_text#'[no_text]'
	return (text_clean)






