import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn import datasets
from sklearn import svm
import itertools
from itertools import cycle
from scipy import interp
from validation import *

def zeros_means(train_values):
	"""Returns the dataframe with the gps_height, longitude, latitude, and population columns edited to replace zeros with mean values"""
	districts = [1,2,3,4,5,6,7,8,13,23,30,33,43,53,60,62,63,67]
	district_values = train_values.filter(['district_code', 'gps_height','longitude','latitude','population'], axis=1)
	district_values = district_values.sort_values(by='district_code')

	#Replace zeros/ones with NaNs for correct calculation of means
	district_values['gps_height'] = district_values['gps_height'].replace(0, np.nan)
	district_values['longitude'] = district_values['longitude'].replace(0, np.nan)
	district_values['latitude'] = district_values['latitude'].replace(0, np.nan)
	district_values['population'] = district_values['population'].replace(0, np.nan)
	district_values['population'] = district_values['population'].replace(1, np.nan)

	#Calculate means
	district_means = district_values.groupby(['district_code']).mean()
	#Round the population column
	district_means['population'] = district_means['population'].round()

	#Replace zeros/ones with NaNs in original dataframe for correct replacements
	train_values['gps_height'] = train_values['gps_height'].replace(0, np.nan)
	train_values['longitude'] = train_values['longitude'].replace(0, np.nan)
	train_values['latitude'] = train_values['latitude'].replace(0, np.nan)
	train_values['population'] = train_values['population'].replace(0, np.nan)
	train_values['population'] = train_values['population'].replace(1, np.nan)

	j = 1;
	for i in districts:
		#Create dictionaries for each district code
		temp_dict = {'gps_height': district_means.iloc[j]['gps_height'], 'longitude': district_means.iloc[j]['longitude'],
					'latitude': district_means.iloc[j]['latitude'], 'population': district_means.iloc[j]['population']}
		#Assign fill values based on dictionary
		train_values.loc[train_values['district_code'] == i, :] = train_values.loc[train_values['district_code'] == i, :].fillna(temp_dict)
		j += 1

	#Districts 0 and 80 have NaN means for gps_height and population, so calculate alternate values and fill in appropriately
	col_means = district_means.mean()
	col_means['population'] = col_means['population'].round()
	dict0 = {'gps_height': col_means['gps_height'], 'longitude': district_means.iloc[0]['longitude'],
					'latitude': district_means.iloc[0]['latitude'], 'population': col_means['population']}

	dict80 = {'gps_height': col_means['gps_height'], 'longitude': district_means.iloc[19]['longitude'],
					'latitude': district_means.iloc[19]['latitude'], 'population': col_means['population']}
	train_values.loc[train_values['district_code'] == 0, :] = train_values.loc[train_values['district_code'] == 0, :].fillna(dict0)
	train_values.loc[train_values['district_code'] == 80, :] = train_values.loc[train_values['district_code'] == 80, :].fillna(dict80)

	return train_values

def zeros_public_meeting(train_values):
	"""Replace zeros in public_meeting column with 'null'"""
	train_values['public_meeting'] = train_values['public_meeting'].replace(0, "null")
	train_values['public_meeting'] = train_values['public_meeting'].replace(np.nan, "null")
	return train_values

def dataframe_tolower(train_values):
	"""Converts strings in dataframe to lowercase"""
	train_values = train_values.applymap(lambda x: x if type(x)!=str else x.lower())
	return train_values

def strings_to_ints(string_col, train_values):
	"""Takes in a list of the string column indices and the main dataframe, returns the dataframe with strings replaced with unique ints. Creates a .csv of the string to int mapping"""
	#NOTE: Make sure csv string columns are entirely strings, ie. don't contain 0's
	unique_list = []

	#Get unique values for each column
	unique_sets = train_values.apply(set)

	#Get unique strings of Training Set columns into one list
	for i in string_col:
		unique_list.extend(list(unique_sets.iloc[i]))

	#Eliminate repeats in between columns
	unique_list = list(set(unique_list))

	#Create a mapping of the strings to their replacement ints
	replace_list = range(len(unique_list))
	replace_list = [k+55000 for k in replace_list]
	string_dict = dict(zip(unique_list, replace_list))

	#Output dictionary to csv
	dictionary = pd.DataFrame.from_dict(string_dict, orient="index")
	dictionary.to_csv("StringDictionary.csv")

	train_values = train_values.applymap(lambda s: string_dict.get(s) if s in string_dict else s)
	return train_values

def strings_to_indicators(train_values):
	"""Extends the given dataframe with a new column for each string in a string column"""
	#Get list of string columns
	columns = [i for i in train_values.columns if type(train_values[i].iloc[0]) == str]
	assert(df_no_nulls(train_values.loc[:, columns]))
	print('Converting the following features to dummies: \n \t')
	pp(columns)
	#For each string column, create a new column for each unique string inside that column
	#and convert the categorical variable into an indicator variable
	for column in columns:
		train_values[column] = train_values[column].replace(0, np.nan) # np.nans are flaots
		new_columns = [column+'_'+str(i) for i in train_values[column].unique()]
		train_values = pd.concat((train_values, pd.get_dummies(train_values[column], prefix = column)[new_columns]), axis = 1)
		del train_values[column]

	return train_values

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def crossval_cmatrices(classifier, num_folds, X_train, Y_labels, class_names):
	"""Generates the overall accuracy with stratified k-fold cross validation and generates confusion matrices for each fold"""
	mean_score = 0
	crossval = StratifiedKFold(n_splits=num_folds, shuffle=False, random_state=None)

	for train, test in crossval.split(X_train, Y_labels):
	    k_pred = classifier.fit(X_train[train], Y_labels[train]).predict(X_train[test])

	    """
	    # Plot normalized confusion matrix for each fold
	    cnf_matrix = confusion_matrix(Y_labels[test], k_pred)
	    np.set_printoptions(precision=1)
	    plt.figure()
	    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True, title='Normalized confusion matrix')
	    plt.show()
	    """

	    # Accuracy of each fold
	    score = accuracy_score(Y_labels[test], k_pred)

	    # To compute mean accuracy over all folds
	    mean_accuracy += accuracy

	mean_accuracy = mean_accuracy / num_folds;
	return mean_accuracy
