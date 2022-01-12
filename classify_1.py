from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier # Class of classifier starts with capital letter

iris = datasets.load_iris()

features = iris.data
labels = iris.target

print(f'features: {features[0]} labels: {labels[0]}') 
# label is giving output 0 because in iris dataset we have 3 categories and each category is given a number 0,1 or 2

# 0 setosa 1 versicolour 2 verginica

# print(iris.DESCR) to print description of the dataset

# Training the classifier
clf = KNeighborsClassifier()
clf.fit(features, labels)

pred = clf.predict([[31,1,1,1]])
print(pred)
