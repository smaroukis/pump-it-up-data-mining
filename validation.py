import numpy as np
import pandas as pd
import string

# This file includes functions that compare dataframes and other objects
# for use in testing and validation

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
    """ Returns a pd.DataFrame that is a diff of two dataframes, pre and post edit"""
    ne_stacked=(df1!=df2).stack()
    changed=ne_stacked[ne_stacked]
    changed.index.names=['id','col']
    diff_loc=np.where(df1!=df2)
    chto=df1.values[diff_loc]
    chfrom=df2.values[diff_loc]
    df = pd.DataFrame({'from':chfrom, 'to':chto}, index=changed.index)
    return df.dropna()

def df_no_nulls(df):
    """ Returns pd.DataFrame of Null and Zero counts of an input dataframe"""
    nulls=df.isnull().sum(axis=0)
    zeros=(df==0).sum(axis=0)
    df_nulls=pd.concat([nulls, zeros], axis=1)
    df_nulls=df_nulls.loc[(df_nulls!=0).any(axis=1)]
    df_nulls.columns=['Null','Zeros']
    print('Checking Null Dataframe...\n')
    if df_nulls.empty:
        print('\t No Nulls or Zeros')
    else:
        print('\t There are Nulls or Zeros')
        print(df_nulls)

    return df_nulls.empty
