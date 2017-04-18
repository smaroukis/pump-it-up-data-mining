from feature_eng import *
from sandy_eng import *

if __name__=="__main__":
    pdb.set_trace()

    # import data
    train_values=load_training_values()
    train_new=deleteColumns(train_values)
    # convert all strings to lowercase
    train_new=dataframe_tolower(train_new)

    # Average over entire column and Replace zeros
    train_new=avgConstrYear(train_new)
    #train_new=zeros_public_meeting(train_new)
    train_new=zeros_permit(train_new)
    train_new=zeros_means(train_new)

    # Replace Empty Strings
    train_new=df_replace_emptystr(train_new)

    # Fuzzy String Matching for Installer, Funder

    #  Create Dummy Columns
    train_new=strings_to_indicators(train_new)

    df_nulls=check_nulls(train_new)
