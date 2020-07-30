# https://www.kaggle.com/rmisra/news-headlines-dataset-for-sarcasm-detection
# nltk.download('popular')
import string
import warnings
import nltk
import re
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
warnings.filterwarnings("ignore")
import pandas as pd
# pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# Read data.
df = pd.read_csv("Data/sentiment_analysis.csv")
print(df.head(20))
print(df.dtypes)
df = df.drop(['airline'], axis=1)

# Check for null values:
print(df.isnull().sum())


# Find the length of Headlines.
def calculate_length(str):
    str_list = str.split()
    return len(str_list)


df['length'] = df['text'].apply(calculate_length)


# Number of unique words.
def unique_words(x):
    return len(set(x.split()))


df['unique_words'] = df.text.apply(unique_words)

# Total number of characters used in a headline.
df['nb_char'] = df.text.apply(lambda x: len(x))


## Number of stopwords per headline.
def stop(text):
    return (len([w for w in text.split() if w in stopwords.words('english')]))


df['nb_stopwords'] = df.text.apply(stop)


############################################
# Text Preprocessing:Text Analytics.
def remove_punctuation(text):
    '''a function for removing punctuation'''
    # replacing the punctuations with no space,
    # which in effect deletes the punctuation marks.
    translator = str.maketrans('', '', string.punctuation)
    # return the text stripped of punctuation marks.
    return text.translate(translator)


def remove_stopwords(text):
    '''a function for removing the stopword'''
    # removing the stop words and lowercasing the selected words
    text = [word.lower() for word in text.split() if word.lower() not in stopwords.words('english')]
    # joining the list of words with space separator.
    return " ".join(text)


def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    tokenization = nltk.word_tokenize(text)
    lemmatized_output = ' '.join([lemmatizer.lemmatize(w) for w in tokenization])
    return lemmatized_output


# Convert text to lowercase.
def tolowercase(text):
    text = text.lower()
    return text


def remove_digits_special_characters(text):
    text = re.sub("[^a-zA-Z]", " ", text)
    return text


# Part-of-speech tagging using NLTK.
def pos_tagging(text):
    text = word_tokenize(text)
    tokens_tag = pos_tag(text)
    return tokens_tag


# Remove twitter handles.
def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)
    return input_txt


#
def remove_short_words(text):
    res = []
    for w in text.split():
        if len(w) > 2:
            res.append(w)
    output = ' '.join(res)
    return output


import numpy as np

# Remove twitter handles (@user).
df['text'] = np.vectorize(remove_pattern)(df['text'], "@[\w]*")
# df['text'] = df['text'].apply(remove_short_words)
df['text'] = df['text'].apply(remove_punctuation)
df['text'] = df['text'].apply(remove_stopwords)
df['text'] = df['text'].apply(lemmatize_text)
df['text'] = df['text'].apply(tolowercase)
df['text'] = df['text'].apply(remove_digits_special_characters)
print(df['text'].head())

# Print unique sentiments.
print(df["airline_sentiment"].unique())

# Print counts of each category.
print(df["airline_sentiment"].value_counts())

# Seperating X and Y.
X = df['text']
Y = df['airline_sentiment']

# Getting training and testing .
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.05, random_state=0)

from sklearn.feature_extraction.text import TfidfVectorizer

word_vectorizer = TfidfVectorizer(max_features=5000)
vector_X_train = word_vectorizer.fit_transform(X_train)
vector_X_test = word_vectorizer.transform(X_test)
# print(word_vectorizer.get_feature_names())
vector_array_train = vector_X_train.toarray()
vector_array_test = vector_X_test.toarray()

# from sklearn.feature_extraction.text import CountVectorizer
# count_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
# # bag-of-words feature matrix.
# vector_X_train = count_vectorizer.fit_transform(X_train)
# vector_X_test = count_vectorizer.transform(X_test)
# # print(word_vectorizer.get_feature_names())
# vector_array_train = vector_X_train.toarray()
# vector_array_test = vector_X_test.toarray()

# # Using linear support vector classifier. # Accepts Sparse Matrix.
# lsvc = LinearSVC()
# lsvc.fit(vector_X_train, y_train)
# y_pred = lsvc.predict(vector_X_test)
# print(f1_score(y_test,y_pred,average='micro')) # 0.77

param_grid = {
     'penalty': ['l1', 'l2'],
     'solver':['newton-cg','lbfgs','liblinear','sag','saga']}
#
from sklearn.model_selection import GridSearchCV
logreg=LogisticRegression()
# Create grid search object.
clf = GridSearchCV(logreg,param_grid = param_grid,cv = 5, verbose=True, n_jobs=-1)
# Fit on data.
clf.fit(vector_array_train, y_train)
print("tuned hyerparameters :(best parameters) ",clf.best_params_)
print("accuracy :",clf.best_score_)
#
# # Logistic Regression.
# lr = LogisticRegression()
# lr.fit(vector_array_train, y_train)
# y_pred = lr.predict(vector_array_test)
# print(f1_score(y_test,y_pred,average='micro'))  # 0.78
#
# # Model 4:-
# # Random Forest Classifier.
# rfc = RandomForestClassifier(n_estimators=10,random_state=0)
# rfc.fit(vector_array_train, y_train)
# y_pred = rfc.predict(vector_array_test)
# print(f1_score(y_test,y_pred,average='micro')) # 0.75

# gbc = GradientBoostingClassifier(n_estimators=15, random_state=10)
# gbc.fit(vector_array_train, y_train)
# y_pred = gbc.predict(vector_array_test)
# print(f1_score(y_test, y_pred, average='micro'))  # 0.66

# svc = SVC(kernel='rbf', C=0.1, gamma='scale')
# svc.fit(vector_array_train, y_train)
# y_pred = svc.predict(vector_array_test)
# print(f1_score(y_test, y_pred, average='micro'))

##########################################
# Working.
# # LDA Topic Modelling:
# https://www.kaggle.com/zikazika/tutorial-on-topic-modelling-lda-nlp
# https://towardsdatascience.com/latent-dirichlet-allocation-for-topic-modelling-explained-algorithm-and-python-scikit-learn-c65a82e7304d
# https://medium.com/analytics-vidhya/topic-modelling-using-latent-dirichlet-allocation-in-scikit-learn-7daf770406c4
from sklearn.feature_extraction.text import CountVectorizer

count_vect = CountVectorizer(max_df=0.8, min_df=2, stop_words='english')
doc_term_matrix = count_vect.fit_transform(X_train)
print(doc_term_matrix)

# Each of x documents is represented as y dimensional vector,which means that our vocabulary has y words.
from sklearn.decomposition import LatentDirichletAllocation
LDA = LatentDirichletAllocation(n_components=5,
                                random_state=0)
print(type(LDA))  # LatentDirichletAllocation class.
LDA.fit(doc_term_matrix)
print(LDA.components_)

# For each topic,each word of the document is assigned a weight.
# Higher weight means it is the top word of the topic.
# It is a multidimensional array.Each row represent the topic,each column represents the word in a document.
# Shape = [n_topics,n_words] or [n_components, n_features]

# Define helper function to print top words for each topic.
def print_top_words(model, feature_names, n_top_words):
    for index, topic in enumerate(model.components_):
        message = "\nTopic #{}:".format(index)
        print(message)
        print(topic)
        print([(feature_names[i], topic[i]) for i in topic.argsort()[:-n_top_words - 1:-1]])
        # feature_names[i] is a word,topic[i] is the weight of the word for that topic.
        print("=" * 70)

number_of_words = 50
print("\nTopics in LDA model: ")
tf_feature_names = count_vect.get_feature_names()
print_top_words(LDA, tf_feature_names, number_of_words)
