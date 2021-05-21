import numpy as np 
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

my_data = pd.read_csv('D:\\drug200.csv', delimiter=",")
# print(my_data[0:5])

########## Pre-processing

X = my_data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values
# print(X[0:5])

# Converting categorical values into numerical values as Sklearn Decision Trees do not handle categorical variables

from sklearn import preprocessing
le_sex = preprocessing.LabelEncoder() 
le_sex.fit(['F','M']) # converting sex values
X[:,1] = le_sex.transform(X[:,1]) 


le_BP = preprocessing.LabelEncoder()
le_BP.fit([ 'LOW', 'NORMAL', 'HIGH']) # converting BP values
X[:,2] = le_BP.transform(X[:,2])


le_Chol = preprocessing.LabelEncoder()
le_Chol.fit([ 'NORMAL', 'HIGH']) # converting cholesterol values
X[:,3] = le_Chol.transform(X[:,3]) 

print(X[0:5])

# Target Y
y = my_data["Drug"]
print(y[0:5])

############# Setting up the Decision Tree

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=3)

############ Modeling

drugTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
print(drugTree) # it shows the default parameters

drugTree.fit(X_train,y_train)

########### Prediction
predTree = drugTree.predict(X_test)

print (predTree [0:5])
print (y_test [0:5])

########## Evaluation

from sklearn import metrics
import matplotlib.pyplot as plt
print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_test, predTree))

########## Visualization

from  io import StringIO
import pydotplus
import matplotlib.image as mpimg
from sklearn import tree
%matplotlib inline 

dot_data = StringIO()
filename = "drugtree.png"
featureNames = my_data.columns[0:5]
out=tree.export_graphviz(drugTree,feature_names=featureNames, out_file=dot_data, class_names= np.unique(y_train), filled=True,  special_characters=True,rotate=False)  
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png(filename)
img = mpimg.imread(filename)
plt.figure(figsize=(10, 20))
plt.imshow(img,interpolation='nearest')


