# Stephen Tucker - PS4 - PCA analysis -  IMT 574 -Spring 23



### Describe the dataset you chose. Why did you choose it? What features does it include? What year is it from? How was it collected? What should we know about this dataset as we read your writeup? (4pts)

#### I choose this dataset for a few reasons, one diabetes typically runs in my family so its of personal interest and me not being in the health care field i wanna know if the PCA analysis accurately identifies the variables that are associated with diabetes. Another reason is the dataset appears to be collected and already clean this saves time in the data processing portion. The variables that exist in 

#### The features in the dataset consist of Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age and Outcome. I assume this was a mixed study with males and females or soley just women based on the pregenancy column. 


### Define a research question. What are you trying to predict? Describe what you're trying to accomplish (it will differ between Supervised and Unsupervised learning). (4pts)

- How well does PCA do in finding variables that cause diabetes? 
- I chose an unsupervised learning technique and the goal is to identify what features have the largest influence on diabetes. 


# Why is this algorithm a good way of answering your research question? (2pts)

- PCA is a good algorithm because it is a great way to save time when you lack domain knowledge in identfying what features play a role in a certain outcome. do to limitations I choose a dataset with a moderate amount of dimensions but in theory this should work for high diminsional data as well.

#### Using the data you chose and the algorithm you chose, read in your data and run your model. (6pts)

# LET THE GAMES BEGIN


```python
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
```


```python
# load data 
data = pd.read_csv("diabetes 2.csv")

```


```python
print(data.head())
```

       Pregnancies  Glucose  BloodPressure  SkinThickness  Insulin   BMI  \
    0            6      148             72             35        0  33.6   
    1            1       85             66             29        0  26.6   
    2            8      183             64              0        0  23.3   
    3            1       89             66             23       94  28.1   
    4            0      137             40             35      168  43.1   
    
       DiabetesPedigreeFunction  Age  Outcome  
    0                     0.627   50        1  
    1                     0.351   31        0  
    2                     0.672   32        1  
    3                     0.167   21        0  
    4                     2.288   33        1  



```python
# Convert the data to a NumPy array
numpy_array = data.to_numpy()
```


```python
numpy_array
```




    array([[  6.   , 148.   ,  72.   , ...,   0.627,  50.   ,   1.   ],
           [  1.   ,  85.   ,  66.   , ...,   0.351,  31.   ,   0.   ],
           [  8.   , 183.   ,  64.   , ...,   0.672,  32.   ,   1.   ],
           ...,
           [  5.   , 121.   ,  72.   , ...,   0.245,  30.   ,   0.   ],
           [  1.   , 126.   ,  60.   , ...,   0.349,  47.   ,   1.   ],
           [  1.   ,  93.   ,  70.   , ...,   0.315,  23.   ,   0.   ]])




```python
pca = PCA(n_components=6)
```


```python
# report findings through variance

print("Explained variance ratio:", pca.explained_variance_ratio_)
print("Principal components:\n", pca.components_)
```

    Explained variance ratio: [0.88853359 0.06159289 0.02578973 0.01308603 0.00744111 0.00302663]
    Principal components:
     [[-2.02174881e-03  9.78118564e-02  1.60930708e-02  6.07566786e-02
       9.93110644e-01  1.40108503e-02  5.37168857e-04 -3.56468123e-03
       5.85325534e-04]
     [-2.26500774e-02 -9.72185778e-01 -1.41901298e-01  5.78559304e-02
       9.46290072e-02 -4.69772538e-02 -8.16920737e-04 -1.40168383e-01
      -7.01033714e-03]
     [-2.24647879e-02  1.43425064e-01 -9.22468119e-01 -3.07012017e-01
       2.09774173e-02 -1.32444012e-01 -6.39970270e-04 -1.25454382e-01
       3.09320895e-04]
     [-4.90410179e-02  1.19804116e-01 -2.62749407e-01  8.84371304e-01
      -6.55494454e-02  1.92810683e-01  2.69930195e-03 -3.01007033e-01
       2.62468355e-03]
     [ 1.51619564e-01 -8.79946326e-02 -2.32159575e-01  2.59926836e-01
      -1.68539731e-04  2.15184772e-02  1.64151800e-03  9.20491753e-01
       6.13101930e-03]
     [-5.07658111e-03  5.08240604e-02  7.55962881e-02  2.21412432e-01
      -6.13718410e-03 -9.70676509e-01 -2.03248123e-03 -1.49790164e-02
      -1.31843876e-02]]


The highest explained variance appears to be at at two components. 

This alogorithim is great when you need to reduce the number of variables in a dataset, it also helps identify which variables are most important.
The biggest indicators of diabetes appear to be columns 2 and 5 Glucose and Insulin. 


```python
X_pca = pca.transform(numpy_array)
```


```python
import matplotlib.pyplot as plt

```


```python
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)')
plt.title('Scree Plot')
plt.show()
```


    
![png](output_18_0.png)
    



```python
X_pca = pca.transform(numpy_array)
```


```python
X_pca
```




    array([[-75.71424916, -35.95494354,  -7.26068338,  15.6705266 ,
             16.50797757],
           [-82.35846646,  28.90955895,  -5.49664901,   9.00443012,
              3.48038132],
           [-74.63022933, -67.90963328,  19.46175322,  -5.65311372,
            -10.29917609],
           ...,
           [ 32.11298721,   3.37922193,  -1.58797191,  -0.87945128,
             -2.98161526],
           [-80.21409513, -14.19059537,  12.35142227, -14.29252832,
              8.53699105],
           [-81.30834662,  21.6230423 ,  -8.15277416,  13.82124771,
             -4.91458328]])




```python
pca = PCA(n_components=7)

```


```python
y_data = data['Outcome']
```


```python
y = 'Outcome'
```


```python
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_data, cmap='coolwarm')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('2D PCA Plot')
```




    Text(0.5, 1.0, '2D PCA Plot')




    
![png](output_24_1.png)
    


I would say this is pretty hard to distinguish betwwen the two outcomes. and in this project it was easier to look at the explained varience then visuals in understanding what. variables influence outcome of diabetes. In building this out ive also considers aspect of scale and best practices dealing with that in showing my visulizations. 

# What challenges did you run into? Do you think it was because of the data, the model, the research question? How would you overcome these challenges? (4pts)


- The biggest challenge i think that can happen with this model is overfitting we can deal with these by leveraging eigenvalue making sure it adds up to about 80% 
- Another challenge is lack of domain expertise. 

#### Ways to overcome 
- adjuist your hyper-parameters to compensate for overfitting. 
- Hire an expert as a consultant. 


#### We learned a little bit about how our models can affect real people in the world. Name 2 potential benefits of your model and 2 potential harms. You can even look at the Wikipedia page on. 


#### Potential harms 
- that exist within my model are the lost of interoperability and information loss. For the latter, this is the cost because of the dimension reduction which is also a benefit so technically it is a cost. 

#### Benefits 
- are dimension reduction, and pattern identification in data this can reveal whats causing a negative outcome. 


### For inspiration. Every model has consequences, what can you think of? If your data is really not amenable to this question, simply write about any other example we covered in class, such as the Boston housing dataset or hate speech detectors. (6pts)


The common consequence that exists with a PCA model can be overfitting, where the model fits the noise in the data rather than the underlying patterns. This is why I mentioned earlier it would be interesting to see how this model would perform with added noise. In particular with the Boston dataset which had problematic mathmatical, processing and documentation errors we as data scientist should ensure we follow up to date best practices and standards. 

### Name one research question you might ask next for future work (don't worry, you don't have to do it!) Why is it important? (2pts)


An interesting research question I wanted to explore was predictive model-based housing data which is leveraging K means, Linear regression, and gradient decent!! It's important to me because I have an interest in the property market as well as related to ethics and the boston data set its important to access these data sets to see the contents and evaluate data collection processes. 

### Report the accuracy of your model. Either through RMSE or another metric. How did accuracy change with your parameter tinkering? (3pts)
- Variance: I changed the number of componets and it alter the varience in that it begun to overfit.

