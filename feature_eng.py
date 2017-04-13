import pandas as pd
import numpy as np
import string
from fuzzywuzzy import fuzz, process
import pdb

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

# Make a function to check unique values of columns:
def check_same(a, b):
    """takes in two iterable objects of the same length and returns the set difference and pct. same"""
    print('{} has {} unique values \n'.format(a.name, len(a.unique())))
    print('{} has {} unique values \n'.format(b.name, len(b.unique())))
    sym_diff=set(a)^set(b)
    print('{} elements in {} or {} but not both: \n {}'.format(len(sym_diff), a.name, b.name, sym_diff))
    check=a==b
    check_false=len(check[check==False])
    print('{pct:.{digits}f}% of elements are the same'.format(pct=100*(1-check_false/len(check)), digits=2))

def col2num(col):
    """turns excel columns A-Z to 0 indexed integers"""
    num=0
    for c in col:
        if c in string.ascii_letters:
            num = num * 26 + (ord(c.upper()) - ord('A'))
    return num

def freq_tab(a,b):
    """Returns a frequency table (divided by rows) of a vs b"""
    ft=pd.crosstab(index=a, columns=b, margins=True)
    return ft/ft.ix['All']

def diff_df(df1,df2):
    ne_stacked=(df1!=df2).stack()
    changed=ne_stacked[ne_stacked]
    changed.index.names=['id','col']
    diff_loc=np.where(df1!=df2)
    chto=df1.values[diff_loc]
    chfrom=df2.values[diff_loc]
    df = pd.DataFrame({'from':chfrom, 'to':chto}, index=changed.index)
    return df.dropna()

def check_nulls(df):
    nulls=df.isnull().sum(axis=0)
    zeros=(df==0).sum(axis=0)
    df_nulls=pd.concat([nulls, zeros], axis=1)
    df_nulls=df_nulls.loc[(df_nulls!=0).any(axis=1)]
    df_nulls.columns(['Null','Zeros'])
    return df_nulls

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
    return df.loc[:, col_list].replace(['','0',0,'-'], 'null')

if __name__=="__main__":
    pdb.set_trace()

    train=load_training_values()
    train=df_replace_emptystr(train)
    merge_installers = create_merge_dict(train,'installer', 79)
    train2=merge_replace(train, merge_installers, 'installer')
    diff = diff_df(train, train2)
    print(diff)
