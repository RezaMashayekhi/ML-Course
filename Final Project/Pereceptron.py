# With the help of assignment 1, 2 in CMPUT 566 Machine Learning Course
import re
import string
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score,precision_score, recall_score
from sklearn.linear_model import Perceptron
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as style
from lime import lime_text
from sklearn.pipeline import make_pipeline

# Reading the dataset files
data = pd.read_csv('train.csv')

# Text Cleaning
def clean_text(text):
    '''
    This function is a cleaning text function which applies many NLP operations
    on the data such as making the text into lower case, removing text in square
    brackets, removing the links, removing punctuation and removeing words
    which cointain numbers.
    '''
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

# Applying the cleaning function to both train and test datasets
data['text'] = data['text'].apply(lambda x: clean_text(x))
# Let's take a look at the updated text
#print(data['text'])


# Vectorize
X = data['text'].values
y = data['target'].values

#changing 0 targets to -1
for i in range(len(y)):
    if y[i]==0:
        y[i]=-1

vectorizer = CountVectorizer(
        lowercase=True, stop_words=None,
        max_df=1.0, min_df=1, max_features=None,  binary=True
      )
X = vectorizer.fit_transform(X).toarray()

# Dividing the train data to two separate parts train and test. (Cross Validation)
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)
print('Train: X=%s, y=%s' % (train_X.shape, train_y.shape))
print('Test: X=%s, y=%s' % (test_X.shape, test_y.shape))

# Adding one feature for bias
train_X=np.insert(train_X, 0, 1, axis=1)
test_X=np.insert(test_X, 0, 1, axis=1)
feature_names = vectorizer.get_feature_names()



class MyPerceptronClassifier():
    def __init__(self, x):
        self.lr = x  #learning rate

    def setLr(self, x):
        self.lr = x

    def train(self, X, y):
        #weights = np.ones(len(feature_names)+1)
        weights = np.random.randint(2,size=len(feature_names)+1)
        for i in range(X.shape[0]):
            if (np.inner(weights,X[i]) * y[i] < 0):
                weights = weights + self.lr * y[i] * X[i]
        return weights

    # for calculating loss function
    def train_Loss(self, X, y, iteration):
        #weights = np.ones(len(feature_names)+1)
        weights = np.random.randint(2,size=len(feature_names)+1)
        List_of_weights=[]
        for k in range(iteration):
            for i in range(X.shape[0]):
                if (np.inner(weights,X[i]) * y[i] < 0):
                    weights = weights + self.lr * y[i] * X[i]
            List_of_weights.append(weights)
        return List_of_weights

    def predict(self, X, weights):
        pred = np.array([])
        for i in range(X.shape[0]):
            if (np.inner(weights,X[i]) > 0):
                pred=np.append(pred,1)
            else:
                pred=np.append(pred,-1)
        return pred

# Finding a good learning rate
max_Learn_Rate=0
max_Accuaracy=0
accuracies1=[]
x_axis=np.arange(0.05,6.0,0.1)
# The first 100 data of test data is considered to be validation set
for i in x_axis:
    clf = MyPerceptronClassifier(i)
    weights=clf.train(train_X, train_y)
    y_pred = clf.predict(test_X[:100], weights)
    accuracy = np.mean((test_y[:100] - y_pred) == 0)
    if(np.mean((test_y[:100] - y_pred) == 0)>max_Accuaracy):
        max_Accuaracy=np.mean((test_y[:100] - y_pred) == 0)
        max_Learn_Rate=i
        max_w=weights
    accuracies1.append(accuracy)

# Learning rate plot
style.use('seaborn-dark') # sets the size of the charts
style.use('ggplot')
plt.plot(x_axis,accuracies1,'steelblue')
plt.xlabel("Learning rate")
plt.ylabel("Accuracy")
print("{} Learning rate has the maximum accuracy {} on our validation set".format(max_Learn_Rate,max_Accuaracy))
plt.show()

# Loss function
clf = MyPerceptronClassifier(max_Learn_Rate)
weights=clf.train_Loss(train_X, train_y, 20)
loss_t=[]
loss_v=[]
for i in range(20):
    y_pred_t = clf.predict(train_X, weights[i])
    loss_t.append(np.mean((train_y - y_pred_t) != 0))
    y_pred_v = clf.predict(test_X[:100], weights[i])
    loss_v.append(np.mean((test_y[:100] - y_pred_v) != 0))
x_axis=np.arange(1,21,1)
plt.plot(x_axis,loss_t,'r',x_axis,loss_v,'b')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(["train", "val"])
plt.show()

# Output
clf = MyPerceptronClassifier(max_Learn_Rate)
total=0
for j in range(20):
    weights=clf.train(train_X, train_y)
    y_pred = clf.predict(test_X[100:], weights)
    total+=np.mean((test_y[100:] - y_pred) == 0)
print(total/20)
print("Average accuracy = {} on test data for {} learning rate in 20 runs with random weights".format(np.mean((test_y[100:] - y_pred) == 0),max_Learn_Rate))
conf_matirx = confusion_matrix(test_y[100:], y_pred)
print("\n\nconf_matrix = \n{}".format(conf_matirx))


# Stack plot
plt.bar("Predicted fake", conf_matirx[1][0], bottom =conf_matirx[0][0], color='darkgray')
plt.bar("Predicted fake", conf_matirx[0][0], color='skyblue')
plt.bar("Predicted real", conf_matirx[0][1], bottom =conf_matirx[1][1], color='darkgray')
plt.bar("Predicted real", conf_matirx[1][1], color='skyblue')
#plt.bar(["predicted fake", "total fakes", "predicted real", "total real"], [np.count_nonzero(y_pred==-1),np.count_nonzero(y_test[100:]==-1),np.count_nonzero(y_pred==1),np.count_nonzero(y_test[100:]==1)])
plt.legend(labels = ['Incorrect prediction','Correct prediction'])
plt.xlabel('Real or fake', fontsize=5)
plt.ylabel('No of tweets', fontsize=5)
#plt.xticks(index, label, fontsize=5, rotation=30)
plt.title('Predicted')
plt.show()
