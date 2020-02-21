# Natural language normalization
import re
from PyBase import Cast




# Normalization of string text.
def StringText(text:str,lower:bool, alphanum:bool)->str:
	try:
		if (lower):     text = text.lower()
		if (alphanum):  text = re.sub('[^0-9a-zA-Z]+', ' ', text)
	except:
		pass
	return(text)

#convert the repositories dictionary in to
# a list of Repository Id and lower case strings.
def DocsText(repo_docs:dict, lower:bool, alphanum:bool)->list:
	repos_text_list: dict = {};
	for repo in repo_docs:
		repos_text_list[repo] = StringText(repo_docs[repo],lower, alphanum);
	return(repos_text_list)


# Create a list of terms for each of the documents.
def DocsTerms(docs_text:dict):
	docs_terms:dict={}
	for doc in docs_text:
		docs_terms[doc] = Cast.StringTerms(docs_text[doc]);
	return(docs_terms);




# # Text Cleaning
# Before we start using the tweets' text we clean it.
# We'll do the this in the class CleanText:
# - remove the **mentions**, as we want to make the model generalisable to tweets of other airline companies too.
# - remove the **hash tag sign** (#) but not the actual tag as this may contain information
# - set all words to **lowercase**
# - remove all **punctuations**, including the question and exclamation marks
# - remove the **urls** as they do not contain useful information and we did not notice a distinction in the number of urls used between the sentiment classes.
# - make sure the converted **emojis** are kept as one word.
# - remove **digits**
# - remove **stopwords**
# - apply the **PorterStemmer** to keep the stem of the words.

# In[5]:
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.base import BaseEstimator
from sklearn.base import  TransformerMixin
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
# **NOTE: **Initially, I made the mistake to do execute TweetsCounts and CleanText in the GridSearchCV below.
# This took too long as it applies these functions each run of the GridSearch.
# It suffices to run them only once.

# In[8]:
def TextCleanSet(df_eda, text_clean, text_name:str):
	data_frame_model = df_eda
	data_frame_model[text_name] = text_clean
	data_frame_model.columns.tolist()
	return(data_frame_model)






# **NOTE: **One side-effect of text cleaning is that some rows do not have any words left in their text.
# For the CountVectorizer and TfIdfVectorizer this does not really pose a problem.
# However, for the Word2Vec algorithm this causes an error.

# There are different strategies that you could apply to deal with these missing values.
# * Remove the complete row, but in a production environment this is not really desirable.
# * Impute the missing value with some placeholder text like *[no_text]*
# * Word2Vec: use the average of all vectors
#
# Here we will impute with a placeholder text '[no_text]'.

# In[7]:
#def EmptyText(text_clean):
#	empty_clean = text_clean == ''
#	print('{} records have no words left after text cleaning'.format(text_clean[empty_clean].count()))
#	text_clean.loc[empty_clean] = '[no_text]'
#	return(text_clean)

# To show how the cleaned text variable will look like, here's a sample.
# In[6]:
#def CleanTextRun(data_frame, text_col, empty_text,out_path:path, show:bool):
def CleanTextRun(data_frame, text_col, empty_text):
	text_clean = TweetsClean().fit_transform(data_frame[text_col])
	text_clean.sample(5)

	empty_clean = text_clean == ''
	print('{} records have no words left after text cleaning'.format(text_clean[empty_clean].count()))
	text_clean.loc[empty_clean] = empty_text#'[no_text]'

#	text_clean = EmptyText(text_clean)
#	if show:
#		ShowFreqWords(text_clean, out_path.joinpath('bar_freq_word.png'))

	return (text_clean)






