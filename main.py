import re
import nltk
import pickle

from sklearn.datasets import load_files
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer

# Importing the Dataset
data = load_files(r"C:\Users\Rosen\Desktop\bug-severity-prediction\data\train")
X, y = data.data, data.target

# Text Preprocessing
documents = []

stemmer = WordNetLemmatizer()

for sen in range(0, len(X)):
    # Remove all the special characters
    document = re.sub(r'\W', ' ', str(X[sen]))
    
    # remove all single characters
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
    
    # Remove single characters from the start
    document = re.sub(r'\^[a-zA-Z]\s+', ' ', document) 
    
    # Substituting multiple spaces with single space
    document = re.sub(r'\s+', ' ', document, flags=re.I)
    
    # Converting to Lowercase
    document = document.lower()
    
    # Lemmatization
    document = document.split()

    document = [stemmer.lemmatize(word) for word in document]
    document = ' '.join(document)
    
    documents.append(document)

# Converting Text to Numbers
# Bag of Words
# from sklearn.feature_extraction.text import CountVectorizer
# vectorizer = CountVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
# X = vectorizer.fit_transform(documents).toarray()
# TFIDF
tfidfconverter = TfidfVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
X = tfidfconverter.fit_transform(documents).toarray()

# Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)

# Training Text Classification Model and Predicting (using Random forest)
classifier = RandomForestClassifier(n_estimators=1000, random_state=0)
classifier.fit(X_train, y_train) 

y_pred = classifier.predict(X_test)

# Evaluating the Model
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))

# Saving and Loading the Model
with open('bug_severity', 'wb') as picklefile:
    pickle.dump(classifier,picklefile)

with open('bug_severity', 'rb') as training_model:
    model = pickle.load(training_model)

y_pred2 = model.predict(X_test)

print('confusion_matrix ', confusion_matrix(y_test, y_pred2))
print('classification_report ', classification_report(y_test, y_pred2))
print('accuracy_score ', accuracy_score(y_test, y_pred2))

print(len(X))
print(len(X_train))
print(len(X_test))

