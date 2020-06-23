import pathlib as path
import numpy as np




# ## Text variable
# To analyze the text variable we create a class **TweetsCounts**.
# In this class we compute some basic statistics on the text variable.
# This class can be used later in a Pipeline, as well.
# * **count_words** : number of words in the tweet
# * **count_mentions** : referrals to other Twitter accounts, which are preceded by a @
# * **count_hashtags** : number of tag words, preceded by a #
# * **count_capital_words** : number of uppercase words, could be used to *"shout"* and express (negative) emotions
# * **count_excl_quest_marks** : number of question or exclamation marks
# * **count_urls** : number of links in the tweet, preceded by http(s)
# * **count_emojis** : number of emoji, which might be a good indication of the sentiment
# In[3]:
import re
import emoji
import pandas
from sklearn.base import BaseEstimator
from sklearn.base import  TransformerMixin
class TweetsCounts(BaseEstimator, TransformerMixin):

	def count_regex(self, pattern, tweet):
		length = len(re.findall(pattern, tweet))
		if(length>300):
			print('Long tweet', tweet)
		return length

	def fit(self, X, y=None, **fit_params):
		# fit method is used when specific operations need to be done on the train data, but not on the test data
		return self

	def transform(self, X, **transform_params)->pandas.DataFrame:
		count_words = X.apply(lambda x: self.count_regex(r'\w+', x))
		count_mentions = X.apply(lambda x: self.count_regex(r'@\w+', x))
		count_hashtags = X.apply(lambda x: self.count_regex(r'#\w+', x))
		count_capital_words = X.apply(lambda x: self.count_regex(r'\b[A-Z]{2,}\b', x))
		count_excl_quest_marks = X.apply(lambda x: self.count_regex(r'!|\?', x))
		count_urls = X.apply(lambda x: self.count_regex(r'http.?://[^\s]+[\s]?', x))

		# We will replace the emoji symbols with a description, which makes using a regex for counting easier
		# Moreover, it will result in having more words in the tweet
		count_emojis = X.apply(lambda x: emoji.demojize(x)).apply(lambda x: self.count_regex(r':[a-z_&]+:', x))
		data_frame = pandas.DataFrame({'count_words': count_words, 'count_mentions': count_mentions, 'count_hashtags': count_hashtags
			                  , 'count_capital_words': count_capital_words, 'count_excl_quest_marks': count_excl_quest_marks
			                  , 'count_urls': count_urls, 'count_emojis': count_emojis})
		return data_frame


def TextCountsRun(data_frame, text_col: str, class_col: str):
	# In[4]:
	text_count = TweetsCounts()
	data_frame_eda = text_count.fit_transform(data_frame[text_col])

	# Add airline_sentiment to data_fame_eda
	data_frame_eda[class_col] = data_frame[class_col]
	return(data_frame_eda)



# So df_model now contains several statistics. However, our vectorizers (see below) will only need the *clean_text* variable.
# The TweetsCounts statistics can be added as such. To specifically select columns, I wrote the class **ColumnExtractor** below.
# This can be used in the Pipeline afterwards.
# In[9]:
from sklearn.base import BaseEstimator
from sklearn.base import  TransformerMixin
class ColumnExtractor(TransformerMixin, BaseEstimator):
	def __init__(self, cols):
		self.cols = cols

	def transform(self, X, **transform_params):
		return X[self.cols]

	def fit(self, X, y=None, **fit_params):
		return self



# Now that we have the cleaned text of the tweets, we can have a look at what are the most frequent words.
# Below we'll show the top 20 words.
#
# **CONCLUSION: **Not surprisingly the most frequent word is *flight*.
# In[29]:
import collections
from sklearn.feature_extraction.text import CountVectorizer
def ShowFreqWords(text_clean, out_file:path):
	cv = CountVectorizer()
	bow = cv.fit_transform(text_clean)
	word_freq = dict(zip(cv.get_feature_names(), np.asarray(bow.sum(axis=0)).ravel()))
	word_counter = collections.Counter(word_freq)
	word_counter_df = pandas.DataFrame(word_counter.most_common(20), columns=['word', 'freq'])
	return word_counter_df;

