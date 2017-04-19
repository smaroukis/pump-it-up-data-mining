import pandas as pd
import numpy as np
import datetime as dt
import string
from fuzzywuzzy import fuzz, process

def load_training_labels():
    return pd.read_csv('data/Training Set Labels.csv')

def load_training_values():
    return pd.read_csv('data/Training Set Values.csv')

def load_test_values():
    return pd.read_csv('data/Test Set Values.csv')

def load_training_merged():
    labels=load_training_labels()
    values=load_training_values()
    return pd.merge(labels, values, how='inner', on='id')

def deleteColumns(df, catList = ['extraction_type_group','extraction_type_class','quantity_group','source_type','quality_group','recorded_by',
              'num_private','payment_type','waterpoint_type_group','scheme_name','amount_tsh', 'region_code', 'public_meeting']):
    """ Function to delete columns from dataframe. Inputs are (1) dataframe, (2) matrix of stringss."""
    for category in catList:
        df = df.drop(category,1)
    return df

def avgConstrYear(df):
    constrMean = df[df['construction_year'] > 0]['construction_year'].mean()
    df['construction_year'] = df['construction_year'].replace(np.nan,constrMean)
    df['construction_year'] = df['construction_year'].replace(0,constrMean)
    return df

def zeros_permit(df):
	"""Replace zeros in public_meeting column with 'Neutral'"""
	df['permit'] = df['permit'].replace(0, "null")
	df['permit'] = df['permit'].replace(np.nan, "null")
	return df

def convert_dates(df, date_col='date_recorded'): 
    """ Given a dataframe and the column referencing date values, return a new dataframe with the dates converted to ordinal"""
    dates=pd.to_datetime(df[date_col])
    ords=dates.apply(lambda x: x.toordinal())
    ords_range=ords.max()-ords.min()
    # Convert to standard normal
    ord_norm=ords.apply(lambda x: (x-ords.mean())/ords.std())
    print('\t Dates have been converted: Descriptive Statistics:')
    print(ord_norm.describe())

    df[date_col]=ord_norm
    return df


def create_merge_dict(df, colname, cutoff):
    """takes data frame and column and creates nested dict of key:[val1, val2...] where val_i is a fuzzy string match with key"""
    _list=[i.lower() for i in df[colname].unique() if ((type(i)!=int) and (type(i)!=float))]
    _list=sorted(_list)
    print("--create_merge_dict()--> creating nested dict of strings to merge\n")
    print("---> Number of items in the list to merge: {} \n".format(len(_list)))
    check_dict={}
    N=len(_list)
    i=iter(range(0,N))
    j=iter(range(10,N+2)) # 2 so we can also get the last element of _list before StopIteration
    while True:
        try:
            jj=next(j)
            ii=next(i)
            check=_list[ii]
            against=_list[ii+1:jj]
            matched={}
            for x in against:
                score=fuzz.token_sort_ratio(check, x)
                if score > cutoff:
                    matched.update({x:score})
            if matched!={}: check_dict.update({check:matched})
        except StopIteration: # j goes off
            # last element at N-1
            while True:
                try:
                    ii=next(i)
                    check=_list[ii]
                    against=_list[ii+1:N] #last elements
                    for x in against:
                        score=fuzz.token_sort_ratio(check, x)
                        if score > cutoff:
                            matched.update({x:score})
                    # if matched!={}: check_dict.update({check:matched})
                except StopIteration: # i goes off, may have empty string comparison at end
                    break
            return check_dict

def merge_replace(_df, colname, merge_dict): # TODO only returns 7 changes
    """takes in a dataframe and a dictionary (see create_merge_dict) and replaces all the list of values with the keys"""
    df=_df
    for i in reversed(sorted(merge_dict)):
        against_list=list(merge_dict[i])
        # find in dataframe
        df[colname].replace(against_list, i, inplace=True)
    print('\t merge_replace(): \n\t ----->  \t Number of Strings Left: {}'.format(len(df[colname].unique())))
    return df

def df_replace_emptystr(df, col_list=['funder','installer']):
    df.loc[:, col_list]=df.loc[:, col_list].replace(['','0',0,'-'], 'null')
    return df

if __name__=="__main__":
    pdb.set_trace()

    train=load_training_values()
    train=df_replace_emptystr(train)
    merge_installers = create_merge_dict(train,'installer', 79)
    train2=merge_replace(train, merge_installers, 'installer')
    diff = diff_df(train, train2)
    print(diff)
