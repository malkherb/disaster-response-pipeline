import sys


def load_data(database_filepath):
    """
    Loads data from SQL Database
    Args:
    database_filepath: The path of the database file
    Returns:
    X: Features dataframe
    Y: Target dataframe
    category_names list: Target labels 
    """
	# import libraries
	import re
	%matplotlib inline
	import matplotlib.pyplot as plt
	import nltk
	nltk.download(['punkt', 'wordnet', 'stopwords'])
	from nltk.corpus import stopwords
	from nltk.tokenize import word_tokenize
	from nltk.stem import WordNetLemmatizer
	import numpy as np
	import pandas as pd
	import pickle
	from nltk.tokenize import word_tokenize
	from nltk.stem import WordNetLemmatizer
	from nltk.corpus import stopwords
	import re
	from sklearn.ensemble import RandomForestClassifier
	from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, precision_recall_fscore_support, label_ranking_average_precision_score
	from sklearn.model_selection import train_test_split, GridSearchCV
	from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
	from sklearn.multioutput import MultiOutputClassifier
	from sklearn.pipeline import Pipeline
	from sklearn.tree import DecisionTreeClassifier
	from sqlalchemy import create_engine
	from nltk.corpus import stopwords
	from nltk.tokenize import word_tokenize
	from nltk.stem.porter import PorterStemmer
	from sklearn.multiclass import OneVsRestClassifier
	from sklearn.svm import LinearSVC
	# load data from database
	engine = create_engine('sqlite:///{}'.format(database_filepath))
	df = pd.read_sql_table('disaster_data', con = engine)
	X = df['message']
	#Limit the rows to speed up the fitting process later:
	X = X[:100]
	y = df.drop(['id', 'message', 'original', 'genre'], axis = 1)
	y = y[:100]
	category_names = y.columns
	return X, y, category_names


def tokenize(text):
    """
    Tokenizes text data
    Args:
    text string: Messages as text data
    Returns:
    words list: Processed text after tokenizing and normalizing the text.
    """
	def tokenize(text):
 
	 #Normalize text
	 norm_words = re.sub(r'[^a-zA-Z0–9]',' ',text)
 
	 #Tokenze words
	 words = word_tokenize(norm_words)
 
	 #Stop words 
	 words = [w for w in words if w not in stopwords.words("english")]
 
	 #Lemmatize
	 lemmed = [WordNetLemmatizer().lemmatize(w, pos="v") for w in words]
 
	 return lemmed


def build_model():
    """
    Builds model with GridSearchCV
    
    Returns:
    Trained model
    """
	pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
							 ('tfidf', TfidfTransformer()),
							 ('clf', MultiOutputClassifier(
								OneVsRestClassifier(LinearSVC())))])

	model = GridSearchCV(estimator=pipeline,
				param_grid=parameters,
				verbose=3,
				cv=3, n_jobs = -1)
	model.fit(X_train,y_train)
	return model

def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluates model's performance on the testing data
    Args:
    model: trained model
    X_test: Test features
    Y_test: Test targets
    category_names: Target labels
    """
	# print classification report
	y_pred = model.predict(X_test)
	print(classification_report(Y_test.values, y_pred, target_names=category_names))

    # print accuracy score
    print('Accuracy: {}'.format(np.mean(Y_test.values == y_pred)))


def save_model(model, model_filepath):
    """
    Saves the model to a Python pickle file    
    Args:
    model: Trained model
    model_filepath: Filepath to save the model
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()