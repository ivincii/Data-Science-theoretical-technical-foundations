```python
# Load the data set
df=pd.read_csv("diabetes_dataset.csv")
df.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pregnancies</th>
      <th>Glucose</th>
      <th>BloodPressure</th>
      <th>SkinThickness</th>
      <th>Insulin</th>
      <th>BMI</th>
      <th>DiabetesPedigreeFunction</th>
      <th>Age</th>
      <th>Outcome</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6</td>
      <td>148</td>
      <td>72</td>
      <td>35</td>
      <td>0</td>
      <td>33.6</td>
      <td>0.627</td>
      <td>50</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>85</td>
      <td>66</td>
      <td>29</td>
      <td>0</td>
      <td>26.6</td>
      <td>0.351</td>
      <td>31</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8</td>
      <td>183</td>
      <td>64</td>
      <td>0</td>
      <td>0</td>
      <td>23.3</td>
      <td>0.672</td>
      <td>32</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>89</td>
      <td>66</td>
      <td>23</td>
      <td>94</td>
      <td>28.1</td>
      <td>0.167</td>
      <td>21</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>137</td>
      <td>40</td>
      <td>35</td>
      <td>168</td>
      <td>43.1</td>
      <td>2.288</td>
      <td>33</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Feature Selection

x = df.drop(['Outcome'], axis=1) # becomes a new dataframe where we drop the outcome column

y = df.Outcome # list solely the values within the outcome column

```


```python
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split

import numpy as np
```


```python
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn import tree

from sklearn.model_selection import train_test_split # Import train_test_split function

from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


from matplotlib import pyplot as plt

```


```python
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
```


```python
# Create Decision Tree classifer object
model = DecisionTreeClassifier()
model

```




    DecisionTreeClassifier()




```python
# Train Decision Tree Classifer
model = model.fit(x_train,y_train)
model
```




    DecisionTreeClassifier()




```python
#Predict the response for test dataset
y_pred = model.predict(x_test)
y_pred
```




    array([0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0,
           0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1,
           0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0,
           0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1,
           0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0,
           1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])




```python
# classification table 
print(classification_report(y_test,y_pred))
```

                  precision    recall  f1-score   support
    
               0       0.75      0.78      0.76        99
               1       0.57      0.53      0.55        55
    
        accuracy                           0.69       154
       macro avg       0.66      0.65      0.65       154
    weighted avg       0.68      0.69      0.69       154
    


define precision: Precision metric is used to evaluate the performence of a machine learning model in a classification task. Specificlly it measures the proportion of true positives predictions among all the positive predictions made by the model.

a high precision score indicates that the model is making few false positive predictions.

precision = true positives / (true positives + false positives)

define recall: is a performence metric used which measures the proportion of true positives predictions amont all the actual positive instances. Recall is important in application where false negatives can have serious consquences. recall = true positives / (true positives + false negatives)

define f1-score: is a performence metric that combines both precision and recall to provide a more comprehensive evaluation of machine learning model. Ranging from 0 - 1 and a higher score indicating better performence.




```python
# Accuracy
print("Accuracy:",metrics.accuracy_score(y_test, y_pred)*100)
```

    Accuracy: 68.83116883116884



```python
# Confusion matrix 
confusion_matrix(y_test,y_pred)
```




    array([[77, 22],
           [26, 29]])



Visualizing A Decision Tree


```python
# Libraries 
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from io import StringIO  

from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
```


```python
dt = DecisionTreeClassifier()
dt.fit(x_train, y_train)
plt.figure(figsize=(10,10))
plot_tree(dt, feature_names=x_train.columns, filled=True)
plt.show()
```


    
![png](output_14_0.png)
    


Some other ways wee could tackle this problem are through random forrest ot Adaboost these stratagies help vary ypur maximum depth. I will attempt a random forrest to compare results. 


```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
```


```python
# training data 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
```


```python
rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
```


```python
# training the random forest on the training data
rf.fit(x_train, y_train)
```




    RandomForestClassifier(max_depth=5, random_state=42)




```python
y_pred = rf.predict(x_test)
y_pred
```




    array([0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
           0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0,
           0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1,
           0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0,
           0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1,
           0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
           0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
           0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0,
           0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0,
           0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1,
           1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])




```python
score = rf.score(x_test, y_test)
score

```




    0.7532467532467533




```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
```


```python
plt.figure(figsize=(10, 10))
plot_tree(rf.estimators_[0], feature_names=x_train.columns, filled=True)
plt.show()
```


    
![png](output_23_0.png)
    



```python

```
