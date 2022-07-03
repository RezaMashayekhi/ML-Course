# Preliminaries
import re
import string
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.optimizers import SGD, Adam
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score,precision_score, recall_score
from tensorflow.keras.utils import to_categorical

plt.style.use('ggplot')

# Reading the dataset files
data = pd.read_csv('train.csv')

# Visualzing the class distribution
#x = data.target.value_counts()
#print(x)
#sns.barplot(x=x.index, y=x, palette='colorblind')
#plt.gca().set_ylabel('Tweets')
#plt.show()

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
vectorizer = CountVectorizer(
        lowercase=True, stop_words=None,
        max_df=1.0, min_df=1, max_features=None,  binary=True
      )
X = vectorizer.fit_transform(X).toarray()
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)
print('Train: X=%s, y=%s' % (train_X.shape, train_y.shape))
print('Test: X=%s, y=%s' % (test_X.shape, test_y.shape))

train_y = to_categorical(train_y)
test_y = to_categorical(test_y)
print('Response: training=%s, testing=%s' % (train_y.shape, test_y.shape))

# %%
hidden_layers = 2
epochs = 10
batch_size = 32
learning_rate = 0.003
dropout_rate = 0.1
input_dimension=16543
output_dimension=2
# %%
DL_Model = Sequential()
DL_Model.add(Dense(1000, input_dim=input_dimension, activation='relu'))
for i in range(hidden_layers):
    DL_Model.add(Dense(500, activation='relu'))
#DL_Model.add(Dropout(dropout_rate))
DL_Model.add(Dense(output_dimension, activation='sigmoid'))
DL_Model.compile(optimizer=Adam(lr=learning_rate),
                 loss='categorical_crossentropy', metrics=['accuracy'])

DL_Model.fit(train_X, train_y, epochs=epochs, batch_size=batch_size, verbose=0)
_, accuracy_ontest = DL_Model.evaluate(test_X, test_y)

print('Accuracy on the test set:', accuracy_ontest)
