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
#		docs_terms[doc] = docs_text[doc].split();
	return(docs_terms);
