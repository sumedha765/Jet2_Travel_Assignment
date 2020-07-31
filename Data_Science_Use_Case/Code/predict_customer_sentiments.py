import string
import warnings
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
warnings.filterwarnings("ignore")
import pandas as pd
pd.set_option('display.max_columns', None)
from sklearn.metrics import f1_score
from sklearn.svm import LinearSVC
from wordcloud import WordCloud,STOPWORDS
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.linear_model import RidgeClassifier,LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.decomposition import LatentDirichletAllocation

# Read data.
df = pd.read_csv("sentiment_analysis.csv")
print(df.head(5))
print(df.dtypes)

##############################################
# Text Preprocessing:Text Analytics.

# Remove twitter handles.
def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)
    return input_txt

# Function to remove punctuation marks.
def remove_punctuation(text):
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)

# Function to remove stopwords.
def remove_stopwords(text):
    result = []
    for word in text.split():
        if word.lower() not in stopwords.words('english'):
            result.append(word.lower())
    return " ".join(result)

# Function to reduce words to their base form:Lemmatization.
def lemmatize_text(text):
    result =[]
    lemmatizer = WordNetLemmatizer()
    tokenization = nltk.word_tokenize(text)
    for w in tokenization:
        result.append(lemmatizer.lemmatize(w))
    return ' '.join(result)

def remove_digits_special_characters(text):
    text = re.sub("[^a-zA-Z]"," ", text)
    return text

# Text preprocessing using various functions.
df['text'] = np.vectorize(remove_pattern)(df['text'], "@[\w]*")
df['text'] = df['text'].apply(remove_punctuation)
df['text'] = df['text'].apply(remove_stopwords)
df['text'] = df['text'].apply(lemmatize_text)
df['text'] = df['text'].apply(remove_digits_special_characters)
print(df['text'].head(10))

###############################################
# Exploratory Data analysis(EDA) :

# Check for null values:
print(df.isnull().sum())

# Print unique categories of sentiments.
print(df["airline_sentiment"].unique())
# Print counts of each category.
print(df["airline_sentiment"].value_counts())

# Print unique airlines.
print(df["airline"].unique())
# Print counts of each airline.
print(df["airline"].value_counts())

# ###################################################
# Visualizations:

# WordCloud for negative words:
negative=df[df['airline_sentiment']=='negative']
words = ' '.join(negative['text'])
cleaned_word = " ".join([word for word in words.split() if 'http' not in word])
wordcloud = WordCloud(stopwords=STOPWORDS,
                      background_color='black',
                      width=3000,
                      height=2500
                     ).generate(cleaned_word)
plt.figure(1,figsize=(12, 12))
plt.imshow(wordcloud)
plt.axis('off')
plt.savefig("negative_words.png")
plt.show()

# WordCloud for positive words:
positive=df[df['airline_sentiment']=='positive']
words = ' '.join(positive['text'])
cleaned_word = " ".join([word for word in words.split() if 'http' not in word])
wordcloud = WordCloud(stopwords=STOPWORDS,
                      background_color='black',
                      width=3000,
                      height=2500
                     ).generate(cleaned_word)
plt.figure(1,figsize=(12, 12))
plt.imshow(wordcloud)
plt.axis('off')
plt.savefig("positive_words.png")
plt.show()

# Visualizing relationship between various sentiments and airlines:
sns.countplot(x='airline',hue='airline_sentiment',data = df)
plt.savefig("Airline_vs_Sentiment.png")
plt.show()

# Visualizing Counts of different airline sentiments.
sns.countplot(x='airline_sentiment',data=df)
plt.savefig("Sentiments.png")
plt.show()

# Visualizing Counts of different airlines.
sns.countplot(x='airline',data=df)
plt.savefig("Airlines.png")
plt.show()

###############
# Encoding categorical target variable.
df['airline_sentiment'] = df['airline_sentiment'].replace({'negative':1,'neutral':0,'positive':2})
###############

# Seperating X and Y.
X = df['text']
Y = df['airline_sentiment']

# Train and Test split.
X_train, X_test, y_train, y_test = train_test_split(X, Y, stratify=Y,test_size=.20,random_state=0)

# Vectorization of words into features:
word_vectorizer = TfidfVectorizer(max_features=5000)
vector_X_train = word_vectorizer.fit_transform(X_train)
vector_X_test = word_vectorizer.transform(X_test)
vector_array_train = vector_X_train.toarray()
vector_array_test = vector_X_test.toarray()

# Modelling:
# Linear support vector classifier.
lsvc = LinearSVC(C=1,loss= 'hinge')
lsvc.fit(vector_array_train, y_train)
y_pred = lsvc.predict(vector_array_test)
print(f1_score(y_test,y_pred,average='micro')) # 0.7872

# Logistic Regression.
lr = LogisticRegression()
lr.fit(vector_array_train, y_train)
y_pred = lr.predict(vector_array_test)
print(f1_score(y_test,y_pred,average='micro'))  # 0.7814

# Random Forest Classifier.
rfc = RandomForestClassifier(n_estimators=100,random_state=0)
rfc.fit(vector_array_train,y_train)
y_pred = rfc.predict(vector_array_test)
print(f1_score(y_test,y_pred,average='micro'))   # 0.7616

# Bernoulli Naive Bayes.
bnb = BernoulliNB()
bnb.fit(vector_array_train,y_train)
y_pred = bnb.predict(vector_array_test)
print(f1_score(y_test,y_pred,average='micro'))  # 0.7790

# Ridge Classifier.
ridge = RidgeClassifier(random_state=0)
ridge.fit(vector_array_train,y_train)
y_pred = ridge.predict(vector_array_test)
print(f1_score(y_test,y_pred,average='micro'))  # 0.7790

##########################################

# Topic Modelling: Understand Themes of customer feedback
# Topic Modelling using Latent Dirichlet Allocation(LDA):
count_vect = CountVectorizer(max_df=0.8, min_df=2,stop_words='english')
doc_term_matrix = count_vect.fit_transform(X_train)

# Each of x documents is represented as y dimensional vector,which means that our vocabulary has y words.
LDA = LatentDirichletAllocation(n_components=5,random_state=0)
LDA.fit(doc_term_matrix)

# For each topic,each word of the document is assigned a weight.
# Higher weight means it is the top word of the topic.
# It is a multidimensional array.Each row represent the topic,each column represents the word in a document.
# Shape = [n_topics,n_words] or [n_components, n_features]

# Print top words for each topic.
def print_top_words(model, feature_names, n_top_words):
    for index, topic in enumerate(model.components_):
        print("\nTopic #{}:".format(index))
        # feature_names[i] is a word,topic[i] is the weight of the word for that topic.
        print([(feature_names[i], topic[i]) for i in topic.argsort()[:-n_top_words - 1:-1]])
        print("=" * 50)

print("\nTopics in LDA model: ")
tf_feature_names = count_vect.get_feature_names()
print_top_words(LDA, tf_feature_names, 10)

#######################