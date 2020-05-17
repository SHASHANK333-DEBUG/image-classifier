# image-classifier

# importing dependencies
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
%matplotlib inline 

# using pandas to read the database stored in the same folder
data =pd.read_csv(r'C:\Users\hp\Downloads\mnist_test.csv\mnist_test.csv')
data.head()

	output:-
  label	1x1	1x2	1x3	1x4	1x5	1x6	1x7	1x8	1x9	...	28x19	28x20	28x21	28x22	28x23	28x24	28x25	28x26	28x27	28x28
0	7	0	0	0	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
1	2	0	0	0	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
2	1	0	0	0	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
3	0	0	0	0	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
4	4	0	0	0	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
5 rows × 785 columns

a=data.iloc[3,1:].values

a = a.reshape(28,28).astype('uint8')
plt.imshow(a)

output:-
<matplotlib.image.AxesImage at 0x125e934f948>

df_x = data.iloc[:,1:]
df_y = data.iloc[:,0]

x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size =0.2, random_state=4)
y_train.head()
output:-
4983    3
6789    1
2221    1
6043    5
1564    7
Name: label, dtype: int64

rf = RandomForestClassifier(n_estimators=100)
rf.fit(x_train, y_train)
output:-
RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                       max_depth=None, max_features='auto', max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=100,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False)

pred = rf.predict(x_test)
pred
output:-
array([1, 3, 7, ..., 0, 1, 0], dtype=int64)

s =y_test.values
count =0
for i in range(len(pred)):
    if pred[i] == s[i]:
        count = count+1
        count
 output:-
 1894
 len(pred)
 output:-
 2000
 1894/2000
 output:-
 0.947

