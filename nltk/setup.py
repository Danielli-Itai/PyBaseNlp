import sys
import nltk

sys.path.append('../../')
from  PyBase import Files

nltk.download (info_or_id="stopwords",download_dir=Files.WorkingDir())
exit(0)