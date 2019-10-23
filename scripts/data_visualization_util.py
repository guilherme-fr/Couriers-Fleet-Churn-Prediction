import pandas as pd

def pairwiseCorr(corrMatrix):
	"""
	Creates a pair wise data frame for better visualization of a correlation matrix.
	
	Parameters
	----------
	corrMatrix : matrix
		Correlation matrix
		
	Returns
	-------
	DataFrame
		Data frame containing 3 columns: 'Attribute_1' holding the one attribute. 'Attribute_2' holding another attribute. 'Value' holding
		the correlation values between the 'Attribute_1' and 'Attribute_2'.
	"""
	
	attr1 = []
	attr2 = []
	value = []
	for rowName, _ in corrMatrix.iterrows():
		for colName, _ in corrMatrix.iteritems():
			if rowName == colName:
				break
			else:
				attr1.append(rowName)
				attr2.append(colName)
				value.append(corrMatrix.loc[rowName, colName])
	pwCorr = pd.DataFrame({'Attribute_1' : attr1, 'Attribute_2' : attr2, 'Value' : value})
	pwCorr = pwCorr.sort_values('Value', ascending = False)
	return pwCorr