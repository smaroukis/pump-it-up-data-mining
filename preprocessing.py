from feature_eng import *
from sandy_eng import *
import ipdb

if __name__=="__main__":
    ipdb.set_trace()
    # Import Data
    train_values=load_training_values()
    labels=load_training_labels()

    train_new=deleteColumns(train_values)
    train_new=dataframe_tolower(train_new)
    train_new=df_replace_emptystr(train_new) # Replace Empty Str

    # Average over entire column and Replace zeros
    train_new=avgConstrYear(train_new) # Fill Zeros (Construction)
    train_new=zeros_permit(train_new) # Fill Zeros (Permit)
    train_new=zeros_means(train_new) # Fill Zeros (GPS)
    train_new=convert_dates(train_new) # Convert Dates to Floats
    cutoff=79
    train_new=fuzzy_string_match(train_new, 'installer', cutoff) # Fuzzy String Matching
    train_new=fuzzy_string_match(train_new, 'funder', cutoff)
    Nlump=20
    train_new=remove_low_frequencies(train_new, Nlump) # Lump Low Occuring

    #  Create Dummy Columns
    train_new=strings_to_indicators(train_new)


    # Classify
    X=train_new.as_matrix()
    y=labels.status_group
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
    rf32 = RandomForestClassifier(criterion='gini',
                                min_samples_split=6,
                                n_estimators=1000,
                                max_features='auto',
                                oob_score=True,
                                random_state=1,
                                n_jobs=-1)

    rf.fit(X_train, y_train.values.ravel())
    OOB=rf32.oob_score_
    print("RF oob score = {}".format(OOB))
