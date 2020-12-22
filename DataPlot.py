import os
import sys
import pandas
import numpy
import pathlib as path
import matplotlib.pyplot as plt


# Set project include path.
sys.path.append('./')
sys.path.append(os.path.join(os.getcwd(),'../'))





# # Exploratory Data Analysis
### Target variable
# There are three class labels to predict: *negative, neutral or positive*.
# **CONCLUSION: **The class labels are **imbalanced** as we can see below.
# This is something that we should keep in mind during the model training phase.
# We could, for instance, make sure the classes are balanced by up/undersampling.
# In[31]:
import seaborn# as sns
seaborn.set(style="darkgrid")
seaborn.set(font_scale=1.3)
def ShowDistribution(data_frame:pandas, class_col:str, file_path:path):
	target_dist = seaborn.catplot(x=class_col, data=data_frame, kind="count", height=6, aspect=1.5, palette="PuBuGn_d")
	plt.show(block=False);
	if file_path:
		target_dist.savefig(file_path)
	plt.close()
	return

# It could be interesting to see how the TextStats statistics relate to the class variable.
# Therefore we write a function **show_dist** that provides descriptive statistics and a plot per target class.
# In[15]:
def show_dist(df, data_col, class_col:str, out_path:path, name:str):
	print('Descriptive stats for {}'.format(data_col))
	print('-' * (len(data_col) + 22))
	print(df.groupby(class_col)[data_col].describe())

	bins = numpy.arange(df[data_col].min(), df[data_col].max() + 1)
	grid = seaborn.FacetGrid(df, col=class_col, height=5, hue=class_col, palette="PuBuGn_d")
	grid = grid.map(seaborn.distplot, data_col, kde=False, norm_hist=True, bins=bins)
	plt.show(block=False)
	grid.savefig(out_path.joinpath(data_col + name))
	plt.close()
	return


def ShowFrequency(word_counter_df:pandas.DataFrame, out_file:path):
	fig, ax = plt.subplots(figsize=(12, 10))
	bar_freq_word = seaborn.barplot(x="word", y="freq", data=word_counter_df, palette="PuBuGn_d", ax=ax)
	plt.show(block=False);
	bar_freq_word.get_figure().savefig(out_file)
	plt.close()
	return;
