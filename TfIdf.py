import sys
import math
from operator import itemgetter





# Convert the repositories dictionary in to
# a list of Repository Id and lower case strings.
def getReposTextList(repo_docs:dict)->list:
	repos_text_list:list = [];
	for repo in repo_docs:
		repo_docs[repo] = [repo_docs[repo][0], repo_docs[repo][1].lower()]
		repos_text_list.append(repo_docs[repo])
	return(repos_text_list)

#Convert the list of document terms in to a unified list of terms.
def getWordset(doc):
	wordSet = []
	for x in doc:
		wordSet = wordSet + x[1]
	return set(wordSet)


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


# Compute the inverse
#https://nlp.stanford.edu/IR-book/html/htmledition/inverse-document-frequency-1.html#fig:figureidf
def CompInvDocFreq(docList)->dict:
	import math

	N = len(docList)
	idfDict = dict.fromkeys(docList[0][1].keys(), 0)
	for doc in docList:
		for word, val in doc[1].items():
			if val > 0:
				idfDict[word] += 1

	for word, val in idfDict.items():
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





class TFIDF():
	#Receives a list of documents, each document is a list of Id and text
	def __init__(self, docs_list:list):
		self.docs_terms = [[x[0], x[1].split()] for x in docs_list]    #Create a list of terms for each documents [id,terms:list].
		dictionary = getWordset(self.docs_terms)                       #return all terms in the dataset (from all documents).
		self.docs_term_freq = CompTermFreq(self.docs_terms, dictionary)     #Term (word) frequency within the whole dataset.
		self.inv_doc_freq = CompInvDocFreq(self.docs_term_freq)            #return tfidf score of each word.
		return;

	#return tfidf vector of each document.
	def getTFIDF(self):
		tfidf_out = {}
		for dic_indx in range(0, len(self.docs_terms)):
			tfidf_out[self.docs_terms[dic_indx][0]] = CompWeighting(self.docs_terms, self.docs_term_freq, dic_indx, self.inv_doc_freq)
		return tfidf_out

	# return tfidf score of each document
	# item[0] doc Id, item[1] sum, item[2] multply.
	def getDocScoring(self, docs_tfidf:list):
		doc_score = [['',0,0] for i in range(len(docs_tfidf))]
		for doc_indx in range(0, len(docs_tfidf)):
			for i in docs_tfidf[doc_indx][1]:
				doc_score[doc_indx][0] = docs_tfidf[doc_indx][0]
				doc_score[doc_indx][1] += docs_tfidf[doc_indx][1][i]
				if(0 == doc_score[doc_indx][2]):
					doc_score[doc_indx][2] = docs_tfidf[doc_indx][1][i]
				else:
					doc_score[doc_indx][2] *= docs_tfidf[doc_indx][1][i]
		return doc_score



	# return similarity score of each document to query
	def search_old(self, tfidf_out:list, query:list):
		query_tfidf = []

		num = 0
		for doc_tfidf in tfidf_out:
			query_tfidf.append(0.0)
			for term in query:
				if term in doc_tfidf:
					query_tfidf[num] += doc_tfidf[term]
			num = num + 1

		hasil = [a * b for a, b in zip(query_tfidf, self.getDocScoring(tfidf_out))]
		for i in range(0, len(hasil)):
			hasil[i] = [i, hasil[i]]

		hasil_akhir = reversed(sorted(hasil, key=itemgetter(1)))
		hasil_akhir = [x for x in hasil_akhir]

		return hasil_akhir

	# return words similarity list for each document.
	def CompFilter(self, docs_tfidf:dict, term_list:list):
		out_tfidf = {}

		for id, doc_weights in docs_tfidf.items():
			query_tfidf = {};
			for doc_trm in doc_weights:
				for term in term_list:
					if(term not in query_tfidf):query_tfidf[term] = 0;
					if term in doc_trm and (doc_weights[doc_trm]>0):
						query_tfidf[term] += doc_weights[doc_trm];#Increase term weight is it is found.

			#Add terms weights.
			out_tfidf[id] = query_tfidf

		return out_tfidf


#Normalize the repositories statistics.
def ScoreNormalize(repos_count:dict, repo_fields:list)->dict:
	#containers initialization.
	fields_stat = {};
	for field in repo_fields:
		fields_stat[field] = {'sum':0,'max':0, 'min':sys.maxsize};

	# Logaritmic reduction for redusing the effect of lage items.
	for repo_id, repo_info  in repos_count.items():
		for field in repo_fields:
			if field in repo_info:
				val:int = repos_count[repo_id][field];
				if(0!=val):
					repos_count[repo_id][field] = math.log10(val)

	# Calculate sum, minimum, maximum.
	for repo_id, repo_info  in repos_count.items():
		for field in repo_fields:
			if field in repo_info:
				fields_stat[field]['sum'] += repo_info[field];
				if fields_stat[field]['max'] < repo_info[field]:
					fields_stat[field]['max'] = repo_info[field]
				if fields_stat[field]['min'] > repo_info[field]:
					fields_stat[field]['min'] = repo_info[field]
			else:
				fields_stat[field] = {'sum':0,'max':0, 'min':0};

	# Normalize the data.
	for repo_id, repo_info  in repos_count.items():
		count = len(repos_count);
		for var in repo_fields:
			if var in repo_info:
				val:int = repos_count[repo_id][var];
				if(fields_stat[var]['sum'] > 0):
					repos_count[repo_id][var] = val / fields_stat[var]['sum'];

	return(repos_count);




######################################################################
#                                                                    #
#              Compute Repositories weights and scores               #
#                                                                    #
######################################################################
def RepoTfIDf(repo_docs:dict):
	repos_text_list = getReposTextList(repo_docs);
	tfidf_alg:TFIDF = TFIDF(repos_text_list)
	docs_tfidf:list = tfidf_alg.getTFIDF();
	docs_score:list = tfidf_alg.getDocScoring(docs_tfidf);
	return(docs_score);

# Repository documents is a list of lists containing [[<doc name>, <doc contents>]]
def RepoTermsTfIDf(repo_docs:dict, terms:list):
	repos_text_list = getReposTextList(repo_docs);
	tfidf_alg:TFIDF = TFIDF(repos_text_list)
	docs_tfidf:dict = tfidf_alg.getTFIDF();

	terms_tfidf = docs_tfidf
	if terms is not None:
		terms_tfidf = tfidf_alg.CompFilter(docs_tfidf, terms);

	tfidf_score = ScoreNormalize(terms_tfidf, terms)
	return(terms_tfidf);
