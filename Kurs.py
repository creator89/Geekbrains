#IMPORT_LIB
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

from sklearn.metrics import r2_score as r2
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error as MSE

from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.manifold import TSNE
plt.style.use('fivethirtyeight')

%config InlineBackend.figure_format = 'svg'
%matplotlib inline

#LOAD_DATA
data = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

#D
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_train_scaled = pd.DataFrame(X_train_scaled, columns = X_train.columns)
X_test_scaled = scaler.transform(X_test)
X_test_scaled = pd.DataFrame(X_test_scaled, columns = X_test.columns)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
feats = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD']
#MODEL_FIT


model = RFR(n_estimators=1000, max_depth=12, random_state=42)
model.fit(X_train, y_train.values[:, 0])
y_pred = model.predict(X_test)
r2(y_test, y_pred)



importances = model.feature_importances_

for feat, imp in zip(feature_names, importances):
    print('{}: {}\n'.format(feat, imp))


#END
test[['Id','Price']].to_csv('AChulkin_predictions.csv',index= None)