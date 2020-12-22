#https://www.geeksforgeeks.org/tokenize-text-using-nltk-python/
import os
import sys
import nltk
from nltk.metrics import scores
from nltk.metrics import confusionmatrix

# Set project include path.
sys.path.append('./')
sys.path.append('../')

from  PyBase import Files





######################################################################
#                                                                    #
#						NLTK API functions 							 #
#                                                                    #
######################################################################

#**************************************************************
#cd c:/Python37
#python -m pip install nltk
#python setup.py
def NltkSetup():
    nltk.download (info_or_id="stopwords",download_dir=Files.WorkingDir())

# splitting the text in to tokens (words).
def Tokenize(text:str):
    return nltk.word_tokenize(text)

# splitting the text in to strings of length size.
def Ngram(text:str,size:int):
    return nltk.ngrams(text, n=size)





# Edit distance score.
# Insertion of a symbol, uv to uxv (ε→x).
# Deletion of a symbol uxv to uv (x→ε).
# Substitution of a symbol uxv to uyv (x→y).
# Returns the number of changes found between the two strings.
def EditDistance(str1:str, str2:str):
    return nltk.edit_distance(str1, str2)

# Returns
# J(X,Y) = |X∩Y| / |X∪Y|
def JacardDistance(str1:str, str2:str):
    w1 = set(str1)
    w2 = set(str2)
    return nltk.jaccard_distance(w1, w2)

# Precision, Recall, F measures.
# Precision - |X∩Y| / |Y| (X=referance, Y=test)
#Recall - |X∩Y| / |X| (X=referance, Y=test)
def PRF_Test(reference_str:str, test_str:str):
    reference_set = set(reference_str)
    test_set = set(test_str)

    test_prec = scores.precision(reference_set, test_set)
    test_recl = scores.recall(reference_set, test_set)
    F= 2 * (test_prec * test_recl) / (test_prec+test_recl)
    return test_prec, test_recl, F

def ConfMatrix(reference_str:str, test_str:str):
    cm = confusionmatrix.ConfusionMatrix(reference_str, test_str)
    return (cm.pretty_format(sort_by_count=True))