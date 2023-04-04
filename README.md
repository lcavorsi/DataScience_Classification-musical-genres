# [DataScience: Classification (musical-genres)](https://github.com/lcavorsi/DataScience_Classification-musical-genres)
Classification model to assign songs to a music genre, according to their technical features <br>
(Part of PGCert in 'Applied Data Science', Birkbeck University)

## Overview

The aim of this assignment is to explore classification models for dividing songs into musical genres. The GTZAN Genre Collection dataset consists of 1000 audio files,  each 30 seconds long, and is divided into 10 music genres, with 100 files each. The dataset's CSV file was downloaded from Kaggle and contains several technical features of the audiofiles.

In this README.md file, not all steps and outputs have been included. These can be checked in the .ipynb file available in the main folder. Here is a summary of the most relevant insights.

Disclaimer: Besides the main aim of classifiyng items, this assignment is also designed to demonstrate how to run a Data Science project, by following all the steps of a data pipeline. <br>
With hindsight, I would have done a few things differently such as probably analysing the data only according to their music genre rather than just on the whole, or becoming technically familiar with the meaning of all the features. <br>
Although the models' accuracy is not very high, I think it represents a good starting point for further developments. Among these, identifying the most critical features to predict music genres could prove particularly fruitful.

## Technology used

- Python 
- KNN Classification model
- Decision Tree Classification model
- Gaussian NB classification model
- KMeans clustering model

## 1: Data loading, checks and cleaning 

- EXTRACTING BASIC DATA. The first step is to familiarise with the dataset and extract basic data (checking that the data have been uploaded correctly, data types, how many values are available for each column, getting basic statistics) <br> 
```
        import pandas as pd #we import the Pandas library to read our dataset as a dataframe 
        df=pd.read_csv('data_GTZAN_coursework.csv') #importing the csv file 
        print(df.dtypes) #visualizing data types per each column (text, integer...) 
        print(df.count()) #visualizing how many values in each columns 
        print(df.describe()) #getting basic statistics 
        print(df.info()) #visualizing datatype, if values are valid, number of entries
        print(df.head()) #visualising the first 5 rows
        print(df.tail()) #visualising the last 5 rows
```


<img src="https://github.com/lcavorsi/DataScience_Classification-musical-genres/blob/main/0_Overview_dataset.JPG" width="300" height="280">

As we can see, the dataset consists of 30 columns, 29 of which associated to 1000 rows and one of which associated to 999 values. Tha majority of data types are float64, as expected, with just one column being integer and two being object. 

- Let's check how many items we have for each musical genre.

```
        print(df.groupby('label').count()) 
```
<img src="https://github.com/lcavorsi/DataScience_Classification-musical-genres/blob/main/1_no.%20of%20items%20per%20genre.JPG" width="600" height="200">

The datasets presents 10 musical genres, with 100 items (audiofiles) per genre.




- STATISTICS. Besides running basic statistcs on the whole dataset, I want to also check the mean of all features by musical genre.

```
        print(df.groupby('label').mean()) 
```

<img src="https://github.com/lcavorsi/DataScience_Classification-musical-genres/blob/main/2_Features%20mean%20per%20musical%20genre.JPG" width="600" height="200">

There are stark differences in the beat and tempo of genres like country and reggae, or between jazz and metal. However, many genres have overlapping values, like classical and blues. The mean of the mfcc features don't seem to reveal any particular or clear pattern either.


- MISSING VALUES. It is crucial to check if there are any missing values.  
```
        print(df.isnull().sum()) 
        print(df[df['rolloff'].isnull()]) #to see the row index of missing values 
```
<img src="https://github.com/lcavorsi/DataScience_Classification-musical-genres/blob/main/3_null%20value%20within%20rolloff.JPG" width="600" height="200">

There is one missing value in the rolloff column, at index 139.

When dealing with missing values, we have different options: leaving it blank, to delete the whole record, to add a value ourselves (we could add the mean if the adjacent values are incremental or adding the missing value by interpolation). When checking if the 'rolloff' values are incremental, I realize they are not. This is confirmed by the scatter plot below.  
   
  
 ```
       df.iloc[135:142] #seeing whether values are sorted incrementally. But they are not

       import matplotlib.pyplot as plt 
       x= df['filename']
       y= df['rolloff']
       plt.scatter(df['x'],df['y'],color='red')
       plt.show()
 ```   
![](https://github.com/lcavorsi/DataScience_Classification-musical-genres/blob/main/4_scatter%20for%20rolloff%20feature.JPG) 


Because values in the rolloff column are not incremental, adding a mean value would not be correct. Adding a missing value by interpolation could be a better alternative. In order to do so, though, a feature strongly related to 'rolloff' must be identified, so as to sort data according to the correlated feature and calculate a potentially good value for the missing one. 
The following heat maps shows a strong correlation between 'rolloff' and 'spectral_centroid'. The correlation value between the two is actually 0.97. 

![](https://github.com/lcavorsi/DataScience_Classification-musical-genres/blob/main/5_HeatMap1.JPG) 
![](https://github.com/lcavorsi/DataScience_Classification-musical-genres/blob/main/6_Heatmap%20for%20rolloff%20feature%20only.JPG) 
![](https://github.com/lcavorsi/DataScience_Classification-musical-genres/blob/main/7_correlation%20value%20between%20rolloff%20and%20centroid.JPG) 

In order to further confirm this insight we plot the rolloff and spectral centoid values. The graph clearly shows a correlation between the two, which mitigates the risks of replacing the missing value with a new one by interpolation.  

![](https://github.com/lcavorsi/DataScience_Classification-musical-genres/blob/main/8_correlation%20plot.JPG) 

Below the data are sorted by spectral centroid (incremental). The missing value is now at index 53. The missing value will ve added through interpolation.

```
df['rolloff'].fillna(df['rolloff'].interpolate(),inplace=True)
```

<img src="https://github.com/lcavorsi/DataScience_Classification-musical-genres/blob/main/9_missing%20value%20sorted%20by%20centroid.JPG" width="800" height="150">
<img src="https://github.com/lcavorsi/DataScience_Classification-musical-genres/blob/main/10_Added%20value%20through%20interpolation.JPG" width="800" height="150">


## 2: Exploring the dataset through visualisations 

- Pairplots to identify skewed features and outliers

Let's visualize at a glance all relationships between variables and their interdependent distribution. This will help us identify skewed variables, hence potential outliers affecting the data mean and the distribution of features.

```
plt.figure() 
sns.pairplot(df)
plt.show() 
```
<img src="https://github.com/lcavorsi/DataScience_Classification-musical-genres/blob/main/11%20scatterplot_all%20variables.JPG" width="500" height="400">

From the pairplot above, we can see that the variables particularly skewed are: rmse, mfcc1, mfcc2, mfcc3, mfcc13. 


Are skewed variables caused by outliers? Visualizing skewed variables through distplots will make any outliers easily recognizable.
```
sns.distplot(df['rmse']) 
sns.distplot(df['mfcc1'])
sns.distplot(df['mfcc2'])
sns.distplot(df['mfcc3'])
sns.distplot(df['mfcc13'])
```

<img src="https://github.com/lcavorsi/DataScience_Classification-musical-genres/blob/main/12_distplot%20skewed%20variables.png" width="900" height="430">

From the distplots of the most skewed variables there doesn't seem to be outliers. <br>
Let's try to visualize the distribution and variability of all features with a boxplot this time. A wide distribution of values could hint to outliers. 

```
graph= sns.boxplot (data=df) 
graph.set_xticklabels(graph.get_xticklabels(),rotation=-90)
```

![](https://github.com/lcavorsi/DataScience_Classification-musical-genres/blob/main/13_Boxplot%20all%20variables.JPG)

Features with the widest distribution are rolloff, spectral bandwidht, spectral centroid. Could this ample distribution of values be determined by outliers? Let's try to visualize these variables via a scatterplot and a kdeplot. Below I'll show the relationship between chroma sfte vs rmse only. This two variables looked particularly skewed, especially in the raggae genre. 

![](https://github.com/lcavorsi/DataScience_Classification-musical-genres/blob/main/14_scatter%20and%20kdeplot%20rmse%20vs%20chroma.png)

While in the scatterplot there seems to be one outlier around 0.40 on the y axis, the kdeplot doesn't display any values too far from the distribution curves.

I decide to analyse outliers by boxplotting features individually. 

![](https://github.com/lcavorsi/DataScience_Classification-musical-genres/blob/main/15_Individual%20boxplots.JPG)

Here, we can see that mfcc1 has a lot of outliers outside 1.5 times the IQR. I therefore decide to select the most prominent outliers (2) and delete them from the dataset
```
df[df.mfcc1<-500] 
df=df.drop([105,176], axis=0)
```

## 3: Preparing the dataset, creating and fitting the models

In this section the dataset will be prepared for model training and fitting. The models will then be evaluated through the train-test-split method and/or the cross-validation method.

- Preparing the dataset (identify features and target)
```
df_features = df[['tempo','beats','chroma_stft','rmse','spectral_centroid','spectral_bandwidth','rolloff','zero_crossing_rate','mfcc1','mfcc2','mfcc3','mfcc4','mfcc5','mfcc6','mfcc7','mfcc8','mfcc9','mfcc10','mfcc11','mfcc12','mfcc13','mfcc14','mfcc15','mfcc16','mfcc17','mfcc18','mfcc19','mfcc20']]
df_target=df[['label']]
X=df_features #X will be our features dataset
y=df_target #y will be our target dataset
```
### KNN Model
- Fitting the KNN model with neighbors value 4
```
from sklearn.neighbors import KNeighborsClassifier
knn= KNeighborsClassifier(n_neighbors=4)
knn.fit(X,y)
```

- Evaluating the KNN model through both the train-test-split and cross-validation method
```
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2,random_state=4) #splitting the dataset: 80% for training, 20% for testing
knn.fit(X_train, y_train) #fitting the model

y_pred=knn.predict(X_test) #running prediction on the test part of the dataset
accuracy=accuracy_score(y_test, y_pred)
print(accuracy)
print(y_test)
print(y_pred)
  
```

```
from sklearn.model_selection import cross_validate
from sklearn import datasets
cross1=cross_validate (knn, X, y, cv=10) 
print(cross1['test_score'].mean())
```

The models' accuracy is respectively 0.39 (train-test-split) and 0.35140404040404044 (cross-validation). 

- Finding the best KNN 

```
import numpy as np
from sklearn.metrics import accuracy_score
k_range=range(1,25)
scores=[]
for k in k_range:
  knn=KNeighborsClassifier(n_neighbors=k)
  knn.fit(X_train, np.ravel(y_train))
  y_pred=knn.predict(X_test)
  accuracy=accuracy_score(y_test,y_pred)
  scores.append(accuracy)
print(scores)
plt.plot(k_range, scores)
plt.xlabel('knn number')
plt.ylabel ('Accuracy') 
```

![](https://github.com/lcavorsi/DataScience_Classification-musical-genres/blob/main/16_best%20knn.JPG)

According to the result, the best performing KNN is 3. When I run the model with KNN 3, the result actually improves to a 0.415 accurancy.

### Decision Tree Model

- Fitting
```
X=df[['tempo','beats','chroma_stft','rmse','spectral_centroid','spectral_bandwidth','rolloff','zero_crossing_rate','mfcc1','mfcc2','mfcc3','mfcc4','mfcc5','mfcc6','mfcc7','mfcc8','mfcc9','mfcc10','mfcc11','mfcc12','mfcc13','mfcc14','mfcc15','mfcc16','mfcc17','mfcc18','mfcc19','mfcc20']] #setting up the features dataaset
y=df[['label']]
from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier()
model.fit(X,y)
model.predict([['69.9374','35','0.402346','0.123468','1855.94','1829.48','3480.94','0.0960399','-177.869','119.197','-17.6007','30.7586','-22.7427','11.9038','-20.7342','3.1806','-9.58348','-0.976488','-11.7763','-5.42061','-9.33936','-9.93932','-3.90989','-5.57062','-1.83902','-2.77842','-3.04687','-8.11581']])
```
- Evaluating through cross-validation
```
from sklearn.model_selection import cross_validate
from sklearn import datasets
cross1=cross_validate (model, X, y, cv=10) 
print(cross1['test_score'].mean())
```

The accuracy for the Decision Tree model is 0.47687878787878785

### GaussianNB model
- Fitting
```
from sklearn.naive_bayes import GaussianNB
modelGaus = GaussianNB()
modelGaus.fit(X,y)
```
- Evaluating through cross-validation
```
from sklearn.model_selection import cross_validate
from sklearn import datasets
cross1=cross_validate (modelGaus, X, y, cv=10) 
print(cross1['test_score'].mean())
```

The accuracy for the GaussianNB model is 0.41868686868686866


### KMeans model

- Fitting the model with 10 clusters
```
cluster_df=pd.read_csv('data_GTZAN_clustering.csv')
from sklearn.cluster import KMeans
modelClust=KMeans (n_clusters=10)
modelClust.fit(cluster_df)

predicted_label=modelClust.predict([['69.9374','35','0.402346','0.123468','1855.94','1829.48','3480.94','0.0960399','-177.869','119.197','-17.6007','30.7586','-22.7427','11.9038','-20.7342','3.1806','-9.58348','-0.976488','-11.7763','-5.42061','-9.33936','-9.93932','-3.90989','-5.57062','-1.83902','-2.77842','-3.04687','-8.11581']])
print(predicted_label)
```
- Visualizing centroids, by plotting only 3 features (tempo, beats, chroma_stft) 

```
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(modelClust.cluster_centers_[:, 1],
            modelClust.cluster_centers_[:, 0],
            modelClust.cluster_centers_[:, 3],
            s = 250,
            marker='o',
            c='red',
            label='centroids')
scatter = ax.scatter(cluster_df['beats'],cluster_df['tempo'], cluster_df['chroma_stft'],
                     c=label,s=20, cmap='winter')


ax.set_title('K-Means Clustering')
ax.set_xlabel('beats')
ax.set_ylabel('tempo')
ax.set_zlabel('chroma_stft')
ax.legend()
plt.show()

```

![](https://github.com/lcavorsi/DataScience_Classification-musical-genres/blob/main/17_K-means%20clustering.JPG)

Unfortunately, the clusters look very close to each other, making the K-means model the least reliable.

## Conclusion

The model which has performed best is the Decision Tree Model. Although the accuracy level is quite low, it lends itself to be possibly considered the best option for further developments in future music genre classification attempts. 
