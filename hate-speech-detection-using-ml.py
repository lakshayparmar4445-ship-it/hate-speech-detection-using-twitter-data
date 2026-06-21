import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
 
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt_tab')  # needed in newer NLTK versions
 
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
 
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV
 
import warnings
warnings.filterwarnings('ignore')
 
print("✅ All libraries imported successfully!")
 
 
# ── STEP 2: Load the dataset ──────────────────────────────────────────────────
 
# BUG FIX: original code used pd.read_python() which does NOT exist.
# Correct function is pd.read_csv() for CSV files.
# Update the path below to where your dataset is saved on your PC.
 
tweet_df = pd.read_csv(r"C:\Users\admin\OneDrive\Desktop\hate speech detection using ML proj internship\train.csv")  # ← change this path if needed
 
print(tweet_df.head())       # shows first 5 rows
print(tweet_df.info())       # shows column names, data types, null counts
 
 
# ── STEP 3: Print some sample tweets ─────────────────────────────────────────
 
print("\n--- Sample Tweets ---")
for i in range(5):
    print(f"Tweet {i+1}:", tweet_df['tweet'].iloc[i], "\n")
 
 
# ── STEP 4: Clean / Preprocess the tweets ────────────────────────────────────
 
# We remove noise: links, hashtags, punctuation, stopwords
# Stopwords = common words like "the", "is", "in" that don't add meaning
 
stop_words = set(stopwords.words('english'))
 
def data_processing(tweet):
    tweet = tweet.lower()                                              # lowercase everything
    tweet = re.sub(r"https\S+|www\S+|http\S+", '', tweet)            # remove URLs
    tweet = re.sub(r'\@\w+|\#', '', tweet)                           # remove @mentions and #
    tweet = re.sub(r'[^\w\s]', '', tweet)                            # remove punctuation
    tweet = re.sub(r'ð', '', tweet)                                   # remove weird symbols
    tweet_tokens = word_tokenize(tweet)                               # split into words
    filtered = [w for w in tweet_tokens if w not in stop_words]      # remove stopwords
    return " ".join(filtered)
 
# Apply the cleaning function to every tweet
tweet_df['tweet'] = tweet_df['tweet'].apply(data_processing)
 
# Remove duplicate tweets
tweet_df = tweet_df.drop_duplicates('tweet')
 
print("✅ Data cleaning done!")
 
 
# ── STEP 5: Lemmatization ─────────────────────────────────────────────────────
 
# Lemmatization = converting words to their root form
# e.g., "running" → "run", "better" → "good"
 
# BUG FIX 1: Original code had a typo — "lemmarizer" instead of "lemmatizer"
# BUG FIX 2: The function was returning 'data' (unchanged) instead of the joined result
 
lemmatizer = WordNetLemmatizer()
 
def lemmatizing(text):
    words = word_tokenize(text)                                        # split text into words
    lemmatized = [lemmatizer.lemmatize(word) for word in words]       # lemmatize each word
    return " ".join(lemmatized)                                        # join back to string
 
tweet_df['tweet'] = tweet_df['tweet'].apply(lemmatizing)
 
print("✅ Lemmatization done!")
print("\n--- Tweets After Preprocessing ---")
for i in range(5):
    print(f"Tweet {i+1}:", tweet_df['tweet'].iloc[i], "\n")
 
 
# ── STEP 6: Check label distribution ─────────────────────────────────────────
 
# label 0 = non-hate tweet
# label 1 = hate tweet
 
print(tweet_df['label'].value_counts())
 
 
# ── STEP 7: Data Visualization ───────────────────────────────────────────────
 
# Bar chart
plt.figure(figsize=(5, 5))
sns.countplot(x='label', data=tweet_df)
plt.title('Count of Hate vs Non-Hate Tweets')
plt.show()
 
# Pie chart
fig = plt.figure(figsize=(7, 7))
colors = ("red", "gold")
wp = {'linewidth': 2, 'edgecolor': "black"}
tags = tweet_df['label'].value_counts()
explode = (0.1, 0.1)
tags.plot(kind='pie', autopct='%1.1f%%', shadow=True,
          colors=colors, startangle=90,
          wedgeprops=wp, explode=explode, label='')
plt.title('Distribution of Sentiments')
plt.show()
 
 
# ── STEP 8: Word Clouds ───────────────────────────────────────────────────────
 
# Word Cloud for NON-HATE tweets
non_hate_tweets = tweet_df[tweet_df['label'] == 0]
text = ' '.join(non_hate_tweets['tweet'])
plt.figure(figsize=(20, 15))
wordcloud = WordCloud(max_words=500, width=1600, height=800).generate(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Most Frequent Words in Non-Hate Tweets', fontsize=19)
plt.show()
 
# Word Cloud for HATE tweets
hate_tweets = tweet_df[tweet_df['label'] == 1]
text = ' '.join(hate_tweets['tweet'])
plt.figure(figsize=(20, 15))
wordcloud = WordCloud(max_words=500, width=1600, height=800).generate(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Most Frequent Words in Hate Tweets', fontsize=19)
plt.show()
 
 
# ── STEP 9: TF-IDF Vectorization ─────────────────────────────────────────────
 
# TF-IDF converts text into numbers so ML model can understand it
# ngram_range=(1,3) means consider single words, pairs, and triplets of words
 
vect = TfidfVectorizer(ngram_range=(1, 3)).fit(tweet_df['tweet'])
 
feature_names = vect.get_feature_names_out()    # BUG FIX: get_feature_names() is deprecated, use get_feature_names_out()
print(f"\nTotal number of features (words/phrases): {len(feature_names)}")
print("First 20 features:", feature_names[:20])
 
 
# ── STEP 10: Prepare X and Y for the model ───────────────────────────────────
 
X = tweet_df['tweet']           # input  (tweets)
Y = tweet_df['label']           # output (0 or 1)
 
X = vect.transform(X)           # convert tweets to TF-IDF vectors
 
# Split into 80% training and 20% testing
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
 
print(f"\nTraining size: {x_train.shape}")
print(f"Testing size:  {x_test.shape}")
 
 
# ── STEP 11: Train Logistic Regression Model ─────────────────────────────────
 
logreg = LogisticRegression()
logreg.fit(x_train, y_train)           # train the model
logreg_predict = logreg.predict(x_test)  # predict on test data
 
logreg_acc = accuracy_score(logreg_predict, y_test)
print(f"\nBasic Logistic Regression Accuracy: {logreg_acc * 100:.2f}%")
 
# Confusion Matrix and Classification Report
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, logreg_predict))
 
print("\nClassification Report:")
print(classification_report(y_test, logreg_predict))
 
# Visual Confusion Matrix
style.use('classic')
cm = confusion_matrix(y_test, logreg_predict, labels=logreg.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=logreg.classes_)
disp.plot()
plt.title("Confusion Matrix - Basic Logistic Regression")
plt.show()
 
 
# ── STEP 12: Hyperparameter Tuning with GridSearchCV ─────────────────────────
 
# GridSearchCV tries many combinations of C and solver
# and picks the best one automatically using 5-fold cross validation
 
param_grid = {
    'C': [100, 10, 1.0, 0.1, 0.01],
    'solver': ['newton-cg', 'lbfgs', 'liblinear']
}
 
grid = GridSearchCV(LogisticRegression(), param_grid, cv=5)
grid.fit(x_train, y_train)
 
print(f"\nBest Cross-Validation Score: {grid.best_score_:.2f}")
print(f"Best Parameters: {grid.best_params_}")
 
 
# ── STEP 13: Evaluate Tuned Model ────────────────────────────────────────────
 
y_pred = grid.predict(x_test)
tuned_acc = accuracy_score(y_pred, y_test)
print(f"\nTuned Model Test Accuracy: {tuned_acc * 100:.2f}%")
 
print("\nConfusion Matrix (Tuned Model):")
print(confusion_matrix(y_test, y_pred))
 
print("\nClassification Report (Tuned Model):")
print(classification_report(y_test, y_pred))
 
# ── DONE ──────────────────────────────────────────────────────────────────────
print("\n✅ Hate Speech Detection project complete!")


