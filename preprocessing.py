from feature_eng import *
from sandy_eng import *
import ipdb

if __name__=="__main__":
    ipdb.set_trace()
    # Import Data
    train_values=load_training_values()

    train_new=deleteColumns(train_values)
    train_new=dataframe_tolower(train_new)
    train_new=df_replace_emptystr(train_new) # Replace Empty Str

    # Average over entire column and Replace zeros
    train_new=avgConstrYear(train_new) # Fill Zeros (Construction)
    train_new=zeros_permit(train_new) # Fill Zeros (Permit)
    train_new=zeros_means(train_new) # Fill Zeros (GPS)
    train_new=convert_dates(train_new) # Convert Dates to Floats
    train_new=fuzzy_string_match(train_new, 'installer') # Fuzzy String Matching
    train_new=fuzzy_string_match(train_new, 'funder')
    train_new=remove_low_frequencies(train_new, 10) # Lump Low Occuring

    #  Create Dummy Columns
    train_new=strings_to_indicators(train_new)
