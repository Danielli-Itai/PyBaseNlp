# pandas is a fast powerful data analysis and manipulation tool built on top of the Python programming language.
# # Loading the data
# We shuffle the data frame in case the classes would be sorted.
# This can be done with the **reindex** method applied on the **permutation** of the original indices.
# In this notebook we will only focus on the text variable and the class variable.
import os
import sys
import pandas
from io import StringIO
import numpy
import pathlib as path


# Set project include path.
sys.path.append('./')
sys.path.append(os.path.join(os.getcwd(),'../'))
from PyBase import Files





######################################################################
#                                                                    #
#							Constants 								 #
#                                                                    #
######################################################################
CSV_SEP_FLD:str = ';'
CSV_SEP_REC:str = '\n'





######################################################################
#                                                                    #
#						Data API functions							 #
#                                                                    #
######################################################################
def DataCsvGet(csv_str:str):
	TESTDATA = StringIO(csv_str)
	data_frame = pandas.read_csv(TESTDATA, engine='python', encoding='utf8')
	return(data_frame)

def DataCsvLoad(file_path:path, scrumbele:bool)->pandas:
	#pandas.set_option('display.max_colwidth', None)
	data_frame = pandas.read_csv(file_path, sep=CSV_SEP_FLD, engine='python',index_col=False, encoding='utf8')
	if scrumbele:
		data_frame = data_frame.reindex(numpy.random.permutation(data_frame.index))
	return data_frame

def DataCsvSave(data_frame:pandas.DataFrame, file_path:path):
	data_frame.to_csv(file_path, sep=CSV_SEP_FLD, index=False)
	return


def rolling(df, window):
	position = 0
	df_length = len(df)
	while position < df_length:
		count = df_length - position
		if count > window: count = window
		yield position, df[position:position + count]
		position += count

def DataCsvSplit(in_path:path, file_name:str, window, out_path:path):
	Files.DirDelete(out_path)
	if not (Files.DirNew(out_path)):  return (None)

	data_file = DataCsvLoad(in_path.joinpath(file_name), False)
	for offset, data_row in rolling(data_file, window):
		DataCsvSave(data_row, out_path.joinpath(file_name + str('%06.1d' % (offset / window))))
	return

def DataCsvMerge(in_path:path, file_name:str, out_path:path, ):
	if not (Files.DirNew(out_path)):  return (None)

	data_file:pandas.DataFrame = pandas.DataFrame()
	for name in Files.FilesDirs(in_path):
		data_csv = DataCsvLoad(name, False)
		if data_file.empty:
			data_file = data_csv
		else:
			data_file = data_file.append(data_csv, ignore_index=True)

	DataCsvSave(data_file, out_path.joinpath(file_name))
	return





