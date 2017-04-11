import pandas as pd
import numpy as np

def zeros_means(train_values):
	"""Returns the dataframe with the gps_height, longitude, latitude, and population columns edited to replace zeros with mean values"""
	
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
	"""Replace zeros in public_meeting column with 'Neutral'"""
	train_values['public_meeting'] = train_values['public_meeting'].replace(0, "Neutral")
	train_values['public_meeting'] = train_values['public_meeting'].replace(np.nan, "Neutral")
	return train_values

def dataframe_tolower(train_values):
	"""Converts strings in dataframe to lowercase"""
	train_values = train_values.applymap(lambda x: x if type(x)!=str else x.lower())
	return train_values
