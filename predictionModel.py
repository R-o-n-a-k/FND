import pandas as pd
from sklearn.model_selection import train_test_split
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
from sklearn.linear_model import PassiveAggressiveClassifier
import itertools
import numpy as np
import seaborn as sb
import pickle

# Load the dataset
df = pd.read_csv("news.csv")

# Function to create distribution plot
def create_distribution(dataFile):
    return sb.countplot(x='label', data=dataFile, palette='hls')

# Checking the distribution of the dataset
create_distribution(df)

# ## Integrity Check(missing value)
def data_qualityCheck():
    print("Checking data qualities...")
    print(df.isnull().sum())
    df.info()
    print("Check finished.")

data_qualityCheck()

# Separate the label
y = df['label']
df.drop('label', axis=1, inplace=True)

# Split dataset for training and testing
X_train, X_test, y_train, y_test = train_test_split(df['text'], y, test_size=0.33, random_state=53)

# Count Vectorizer (Bag of Words)
count_vectorizer = CountVectorizer(stop_words='english')

# Fit and transform the training data
count_train = count_vectorizer.fit_transform(X_train)

# Function to get statistics for Count Vectorizer
def get_countVectorizer_stats():
    print(count_train.shape)  # vocab size
    print(count_vectorizer.vocabulary_)  # check vocabulary

get_countVectorizer_stats()

count_test = count_vectorizer.transform(X_test)

# TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

# Fit and transform the training data for TF-IDF
tfidf_train = tfidf_vectorizer.fit_transform(X_train)

# Function to get statistics for TF-IDF Vectorizer
def get_tfidf_stats():
    print(tfidf_train.shape)
    print(tfidf_train.A[:10])  # Preview of the TF-IDF train data

get_tfidf_stats()

tfidf_test = tfidf_vectorizer.transform(X_test)

# Display the feature names
print(tfidf_vectorizer.get_feature_names_out()[-10:])
print(count_vectorizer.get_feature_names_out()[:10])

# Confusion Matrix plot function
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Naive Bayes Classifier using Count Vectorizer
nb_pipeline = Pipeline([
    ('NBTV', tfidf_vectorizer),
    ('nb_clf', MultinomialNB())
])

# Fit Naive Bayes classifier
nb_pipeline.fit(X_train, y_train)

# Perform classification on test data
predicted_nbt = nb_pipeline.predict(X_test)

# Accuracy and Confusion Matrix
score = metrics.accuracy_score(y_test, predicted_nbt)
print(f'Accuracy: {round(score * 100, 2)}%')
cm = metrics.confusion_matrix(y_test, predicted_nbt, labels=['FAKE', 'REAL'])
plot_confusion_matrix(cm, classes=['FAKE', 'REAL'])
print(cm)

# Passive Aggressive Classifier
linear_clf = Pipeline([
    ('linear', tfidf_vectorizer),
    ('pa_clf', PassiveAggressiveClassifier(max_iter=50))
])
linear_clf.fit(X_train, y_train)

# Predict and calculate accuracy for Passive Aggressive Classifier
pred = linear_clf.predict(X_test)
score = metrics.accuracy_score(y_test, pred)
print(f'Accuracy: {round(score * 100, 2)}%')
cm = metrics.confusion_matrix(y_test, pred, labels=['FAKE', 'REAL'])
plot_confusion_matrix(cm, classes=['FAKE', 'REAL'])

# Save the best model
model_file = 'finalmodel.sav'
pickle.dump(linear_clf, open(model_file, 'wb'))

# Function to make predictions
def detecting_fake_news(var):
    load_model = pickle.load(open('finalmodel.sav', 'rb'))
    prediction = load_model.predict([var])
    print(f"The given statement is {prediction[0]}")

if __name__ == '__main__':
    var = input("Enter the news text you want to verify: ")
    detecting_fake_news(var)
