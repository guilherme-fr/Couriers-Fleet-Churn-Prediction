import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def labelCourier(courierIds, couriersWeeklyData, churn_window):
	"""
	Labels couriers as 'CHURNED' (1) and 'NOT CHURNED' (0).
	
	Parameters
	----------
	courierIds : array
		Array containing the courier's ids to be labeled.
	couriersWeeklyData: DataFrame
		Data frame containing the weekly activity of the couriers.
	churn_window : array
		Array containing the weeks numbers to be considered as the churn window for the labeling.
		
	Returns
	-------
	DataFrame
		Data frame with the following columns: 'courier' holding the courier's ids. 'churned' holding the label '0' (NOT CHURNED) or '1' (CHURNED)
	"""
	
	cwd = couriersWeeklyData
	courierLabel = []
	for cId in courierIds:
		courierWeeks = cwd.loc[cwd['courier'] == cId, 'week']
		if courierWeeks.isin( churn_window ).sum() > 0:
			#If the courier had any activity in the churn weeks
			courierLabel.append(0)
		else:
			#If the courier didn't have any activity in the churn weeks
			courierLabel.append(1)
	
	return pd.DataFrame({'courier': courierIds, 'churned': courierLabel})

def remove_weeks(couriersWeeklyData, weeks_to_remove):
	"""
	Removes from the couriers weekly activities data frame all the rows from the specified weeks.
	
	Parameters
	----------
	couriersWeeklyData : DataFrame
		Data frame containing the weekly activity of the couriers.
	weeks_to_remove : array
		Array containing the number of the weeks to remove from the activity data.
	
	Returns
	-------
	DataFrame
		Couriers weekly activities data without the specified weeks.
	"""
    
	data = couriersWeeklyData.drop(couriersWeeklyData[ couriersWeeklyData['week'].isin(weeks_to_remove) ].index)
	return data
	
def add_churn_col(data_frame, label_courier, churned_value, not_churned_value):
	"""
	Add 'churn' label column to a data frame.
	
	Parameters
	----------
	data_frame : DataFrame
		Data frame where the 'churn' label column will be added. The data frame needs to have a column named 'courier' 
		containing the couriers ids.
	label_courier : DataFrame
			Data frame containing the column 'courier' with the couriers ids, and the column 'churn'.
	
	Returns
	-------
	DataFrame
		The data frame passed as parameter with the 'churn' label column added.
	"""
	
	churned = label_courier[label_courier.churned == 1].courier
	
	churn_col = np.where(data_frame['courier'].isin(churned), churned_value, not_churned_value)
	data = pd.DataFrame(data_frame, index = data_frame.index)
	data['churned'] = churn_col
	
	return data
	
def weeksWorked(couriersWeeklyData):
	"""
	Calculates how many weeks each courier has worked.
	
	Parameters
	----------
	couriersWeeklyData : DataFrame
		Data frame containing the weekly activity of the couriers.
	
	Returns
	-------
	DataFrame
		Data frame containing the column 'courier' with the couriers ids, and the column 'weeks_worked' with 
		the total weeks worked for each courier.
	"""

	n_weeks_worked = couriersWeeklyData[['week']].groupby(couriersWeeklyData.courier).count().reset_index()
	n_weeks_worked.rename(columns = {'week':'weeks_worked'}, inplace = True)
	return n_weeks_worked
	
def create_life_feature1_col(courier_life_data, courier_weekly_data):
	"""
	Takes the 'feature_1' column from the courier's life features data and put together with the courier ids in the
	courier's weekly activities data.
	
	Parameters
	----------
	courier_life_data : DataFrame
		Data frame containing life features data of the couriers.
	courier_weekly_data : DataFrame
		Data frame containing weekly activities data of the couriers.
		
	Returns
	-------
	DataFrame
		Data frame containing the column 'courier' with the couriers ids, and the column 'life_feature1' with 
		the 'feature_1' column from the life features data.
	"""
	
	col_life_feature1 = []
	for courier_id in courier_weekly_data['courier']:
		life_feature1 = courier_life_data.loc[courier_life_data.index == courier_id, 'feature_1']
		col_life_feature1.append(life_feature1.item())
	
	return pd.DataFrame({'courier' : courier_weekly_data['courier'], 'life_feature1' : col_life_feature1})
	
def summarize_data(courier_weekly_data):
	"""
	Summarize all the features in the data frame by sum grouped by each courier.
	
	Parameters
	----------
	courier_weekly_data : DataFrame
		Data frame containing weekly activities data of the couriers.
	
	Returns
	-------
	DataFrame
		Data frame summarized by sum grouped by each courier.
	"""
	summ_data = courier_weekly_data.groupby(courier_weekly_data.courier).sum().reset_index()
	return summ_data
	
def prepare_data_for_modeling(courier_life_data, courier_weekly_data, weeks_to_remove, churn_window):
	"""
	Prepares the data for the classification prediction (CHURNED or NOT CHURNED).
	
	Parameters
	----------
	courier_life_data : DataFrame
		Data frame containing life features data of the couriers.
	courier_weekly_data : DataFrame
		Data frame containing weekly activities data of the couriers.
	weeks_to_remove : array
		Array containing the number of the weeks to remove from the activity data.
	churn_window : array
		Array containing the weeks numbers to be considered as the churn window for the labeling.
	
	Returns
	-------
	DataFrame
		Data frame with summarized data for classification algorithm.
	"""
	
	#Creating the label column to be added at the end
	labels = labelCourier(courier_life_data.index, courier_weekly_data, churn_window)
	
	#Removing the weeks not relevant for the prediction
	data = remove_weeks(courier_weekly_data, weeks_to_remove)
	
	#Calculating the weeks worked by each courier
	weeks_worked = weeksWorked(data)
	
	#Summarizing data
	data = summarize_data(data)
	
	#Adding the new 'weeks_worked' column
	data['weeks_worked'] = weeks_worked['weeks_worked'] 
	
	#Removing the feature 'week' since it doesn't make sense summarized
	data.drop(columns = 'week', inplace = True)
	
	life_feature1_col = create_life_feature1_col(courier_life_data, data)
	#Changing the character values to numeric values
	life_feature1_encoder = LabelEncoder()
	life_feature1_encoder.fit(life_feature1_col['life_feature1'])
	life_feature1_col['life_feature1'] = life_feature1_encoder.transform(life_feature1_col['life_feature1'])
	
	#Adding the 'feature_1' column from 'courier_life_data' to data frame
	data['life_feature_1'] = life_feature1_col['life_feature1']
	
	#Adding the label column to the data frame
	data = add_churn_col(data, labels, 1, 0)
	
	return data