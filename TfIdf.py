import os
import sys
import math

sys.path.append(os.path.join(os.getcwd(),'../'))
from PyBaseNlp import TextNorm





######################################################################
#                                                                    #
#             Term frequency inverse document frequency              #
#                                                                    #
######################################################################

# Convert the repositories dictionary in to
# a list of Repository Id and lower case strings.
def DocsDicToList(repo_docs:dict)->list:
	repos_text_list:list = [];
	for repo in repo_docs:
		repo_docs[repo] = [repo_docs[repo][0], repo_docs[repo][1].lower()]
		repos_text_list.append(repo_docs[repo])
	return(repos_text_list)


# Compute term frequence for each document.
# Return the words dictionary with term occurrences for each document.
def CompTermFreq(docs_list, wordSet)->list:
	termFreq = []
	for i in range(0, len(docs_list)):
		doc_name = docs_list[i][0]
		termFreq.append([doc_name, dict.fromkeys(wordSet, 0)])   # Create a list of document terms with count 0.

	for dic_indx in range(0, len(termFreq)):                    # loop threw documents.
		for word in docs_list[dic_indx][1]:                      # Loop tokens in document.
			termFreq[dic_indx][1][word] += 1                      # Increase the number of occurrences for each term.
	return termFreq


# Compute the inverse document frequency for each vocabulary term.
#https://nlp.stanford.edu/IR-book/html/htmledition/inverse-document-frequency-1.html#fig:figureidf
def CompInvDocFreq(docList)->dict:
	import math

	N = len(docList)												#number of dictionaries.
	idfDict = dict.fromkeys(docList[0][1].keys(), 0)	#Vector of dictionary items.
	for doc in docList:
		for word, val in doc[1].items():
			if val > 0:
				idfDict[word] += val								# count occurances if each term on each document.

	for word, val in idfDict.items():						# compute inverse document frequency.
		idfDict[word] = math.log10(N / float(val))

	return idfDict


#multiply each document term frequency with its inverse document frequaency.
def CompWeighting(docs_terms:list, docs_list:list, dic_indx:int, idfs:dict):
	tfidf = {}
	for word in docs_terms[dic_indx][1]:
		tfidf[word] = 0
		tf = docs_list[dic_indx][1][word]   #term frequency.
		if (tf>0):		tfidf[word] = (1+math.log10(tf)) * idfs[word]
	return tfidf





######################################################################
#                                                                    #
#             Term frequency inverse document frequency              #
#                                                                    #
######################################################################

class TFIDF():
	# Receives a dictionary of documents
	# Each document is a string indexed by the repository id.
	def __init__(self, docs_norm: dict):
		self.docs_norm = docs_norm
		self.docs_vocab = TextNorm.DocsWordset(docs_norm)							#return all terms in the dataset (from all documents).
		self.docs_term_freq = TextNorm.DocsTermFreq(docs_norm, self.docs_vocab)

		self.docs_vocab_freq = TextNorm.VocabWordfrq(self.docs_term_freq, self.docs_vocab)
		self.docs_vocab_inv_freq = TextNorm.VocabInvDocfrqs(len(docs_norm), self.docs_vocab_freq)
		return;

	def getVocabulary(self):
		return self.docs_vocab

	def getDocs(self):
		return self.docs_norm.keys()

	#return tfidf vector of each document.
	def getTFIDF(self):
		tfidf_out = {}
		for key, doc  in self.docs_term_freq.items():
			tfidf_out[key] = {term: doc[term] * self.docs_vocab_inv_freq[term] for term in self.docs_vocab_inv_freq}
		return tfidf_out


	# Calculate tfid similarity acording to query sum.
	def QueryTfidfSim(self, docs_tfidf:dict, filter_words:dict):
		docs_tfidf:dict = self.getTFIDF()
		filter_norm = TextNorm.DocsNorm(filter_words, True, True)				# Normalize the query.
		filter_vocab = TextNorm.DocsWordset(filter_norm)							# Get the filter vocabulary.

		query_scor = dict.fromkeys(docs_tfidf.keys(), 0)
		for key, doc in docs_tfidf.items():
			for term in filter_vocab:
				if term in doc:
					query_scor[key] += doc[term]

		doc_score = {k: v for k, v in sorted(query_scor.items(), key=lambda item: item[1], reverse=True)}  # sort the scores acceedig
		return doc_score


	# Calculate a query TfIdf score for each document.
	def QueryCosinSim(self, docs_tfidf:dict, filter_docs:dict):
		filter_vocab = TextNorm.DocsWordset(filter_docs)							# Get the filter vocabulary.
		filter_term_freq = TextNorm.DocsTermFreq(filter_docs, filter_vocab)	# Get the vocabulary terms frequency.

		# Add docs vocabulary terms to filter frequency vector.
		for filter in filter_term_freq:													# Add query terms to the documents TfIdf.
			for term in self.docs_vocab:													# for each query term.
				if not term in filter_term_freq[filter]:								# if not already in the document
					filter_term_freq[filter][term] = 0									# add to the document vector.

		# Add filter new terms to docs_tfidef vectors.
		for term in filter_vocab:														# for each filter term.
			for key in docs_tfidf:														# for all documents.
				if not term in docs_tfidf[key]:
					docs_tfidf[key][term] = 0											# add filter words to docs vocabulary.

		filter_vocab = self.docs_vocab.union(filter_vocab)						# add documents vocabulary to the filter vocabulary.
		filter_vocab_freq = TextNorm.VocabWordfrq(filter_term_freq, filter_vocab)	#	Calculate the filter vocabulary frequency.

		doc_score = {}
		for key,doc_ifd in docs_tfidf.items():											# Calculate each document similarity score.
			doc_score[key] = TextNorm.QueryCosSim(list(filter_vocab_freq.values()), list(doc_ifd.values()))
		doc_score = {k: v for k, v in sorted(doc_score.items(), key=lambda item: item[1], reverse=True)}	#sort the scores acceedig
		return doc_score





######################################################################
#                                                                    #
#              Compute Repositories weights and scores               #
#                                                                    #
######################################################################
COS_SIM = 'cos_sim'
TFIDF_SIM = 'tfidf_sim'

def FilterDocs(filter_words:list)->dict:
	index:int = 0
	filter_docs = {}
	for filter in filter_words:
		filter_docs['filer' + str(index)] = filter
		index += 1
	return filter_docs

# Repository documents is a list of lists containing [[<doc name>, <doc contents>]]
def DocsTextTfIDf(text_docs:dict):
	docs_norm = TextNorm.DocsNorm(text_docs, True, True)

	tfidf_alg:TFIDF = TFIDF(docs_norm)
	docs_tfidf:dict = tfidf_alg.getTFIDF();

	return(docs_tfidf, tfidf_alg);

# Repository documents is a list of lists containing [[<doc name>, <doc contents>]]
def RepoTermsSimilarity(repo_docs:dict, filter_words:list):
	tfidf_alg:TFIDF = TFIDF(repo_docs)
#	docs_tfidf:dict = tfidf_alg.getTFIDF();

	filter_docs = FilterDocs(filter_words)
	query_score = tfidf_alg.QueryTfidfSim(filter_docs)
	docs_cos_sim = tfidf_alg.QueryCosinSim(filter_docs);
	return docs_cos_sim;
