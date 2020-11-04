#https://www.geeksforgeeks.org/tokenize-text-using-nltk-python/

import sys
import nltk

sys.path.append('../')
from  PyBase import Files




#**************************************************************
#cd c:/Python37
#python -m pip install nltk
#python setup.py
def NltkSetup():
    nltk.download (info_or_id="stopwords",download_dir=Files.WorkingDir())

