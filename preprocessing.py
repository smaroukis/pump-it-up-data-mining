from feature_eng import *
from sandy_eng import *
import ipdb

if __name__=="__main__":
    ipdb.set_trace()
    # Import Data
    train_values=load_training_values()
    train_new=deleteColumns(train_values)
    # Conver All Strings to Lowercase
    train_new=dataframe_tolower(train_new)

    # Average over entire column and Replace zeros
    train_new=avgConstrYear(train_new)
    train_new=zeros_public_meeting(train_new)
    train_new=zeros_permit(train_new)
    train_new=zeros_means(train_new)

    # Replace Empty Strings
    train_new=df_replace_emptystr(train_new)

    # Fuzzy String Matching for Installer, Funder
    # TODO

    # Lump Low Frequencies
    train_new=remove_low_frequencies(train_new, 10)

    # Convert Dates to floats with std normal distribution
    train_new=convert_dates(train_new)  # Takes awhile

    #  Create Dummy Columns
    train_new=strings_to_indicators(train_new)
