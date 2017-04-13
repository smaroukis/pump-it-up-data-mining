from feature_eng import *
from sandy_eng import *
from noah_eng import *

if __name__=="__main__":

    # import data
    train_values=load_train_values()
    train_new=deleteColumns(train_values)
    # convert all strings to lowercase
    train_new=dataframe_tolower(train_new)

    # Average over entire column and Replace zeros
    train_new=avgConstrYear(train_new)
    train_new=zeros_public_meeting(train_new)
    train_new=zeros_permit(train_new)
    train_new=zeros_means(train_new)

    # Replace Empty Strings
    train_new=df_replace_emptystr(train_new)

    # Fuzzy String Matching for Installer, Funder

    #  Create Dummy Columns

    pass
