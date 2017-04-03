import pandas as pd

#NOTE: Make sure csv string columns are entirely strings, ie. don't contain 0's
# 	   and edit the variable string_col to reflect these columns

unique_list = []
string_col = [1,2] 	#Indexes of string columns in training csv

#Read in csv
train_values = pd.read_csv('Training Set Values.csv')

#Convert all strings to lowercase
train_values_lower = train_values.applymap(lambda x: x if type(x)!=str else x.lower())

#Get unique values for each column
unique_sets = train_values_lower.apply(set)

#Get unique strings of Training Set columns into one list
for i in string_col:
	unique_list.extend(list(unique_sets.iloc[i]))

#Sort the list
unique_list.sort()

#Eliminate repeats in between columns
unique_list = list(set(unique_list))

#Create a mapping of the strings to their replacement ints
replace_list = range(len(unique_list))
string_dict = dict(zip(unique_list, replace_list))

#Output dictionary to csv
dictionary = pd.DataFrame.from_dict(string_dict, orient="index")
dictionary.to_csv("StringDictionary.csv")

#Output new training values to csv
train_ints = train_values_lower.applymap(lambda s: string_dict.get(s) if s in string_dict else s)
train_ints.to_csv('TrainingSetInts.csv')