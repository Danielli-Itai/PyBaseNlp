# Natural language normalization
import re
from PyBase import Cast




# Normalization of string text.
def StringText(text:str,lower:bool, alphanum:bool)->str:
	try:
		if (lower):
				text = text.lower()
		if (alphanum):
			text = re.sub('[^0-9a-zA-Z]+', ' ', text)
	except:
		pass
	return(text)

#convert the repositories dictionary in to
# a list of Repository Id and lower case strings.
def DocsText(docs_list:dict, lower:bool, alphanum:bool)->dict:
	docs_text = {}
	for doc_naeme in docs_list:
		text = StringText(docs_list[doc_naeme][1], lower, alphanum)
		docs_text[doc_naeme] = [doc_naeme,text]
	return docs_text

def DocsStr(docs_list:dict, lower:bool, alphanum:bool)->str:
	docs_text = ''
	for doc_naeme in docs_list:
		text = StringText(docs_list[doc_naeme][1], lower, alphanum)
		docs_text += text
	return docs_text



# Create a list of terms for each of the documents.
def DocsTerms(docs_text:dict):
	docs_terms:dict={}
	for doc in docs_text:
		docs_terms[doc] = Cast.StringTerms(docs_text[doc]);
	return(docs_terms);

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






