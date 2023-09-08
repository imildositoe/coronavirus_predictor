import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures

#load data
data = pd.read_csv('corona_cases.csv')
data = data[['id', 'cases']]
print('-------------------------HEAD----------------------')
print(data.head())


#prepare data
print('-------------------------PREPARE DATA----------------------')
#converting the data (x, y) into numpy arrays
x = np.array(data['id']).reshape(-1, 1)
y = np.array(data['cases']).reshape(-1, 1)
#ploting to a  visual graph
plt.plot(y, '-m')
plt.show()

#adding more powered columns
polyFeart = PolynomialFeatures(degree=2)
x = polyFeart.fit_transform(x)
print(x)

