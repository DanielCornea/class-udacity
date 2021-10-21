# import libraries
import sys
import pandas as pd 
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import ExtraTreesClassifier
import joblib
import nltk
import re
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


def load_data(database_filepath):
    """Loads data from database
       
       Args:  

           database_filepath (string): the filepath to the database

       Returns:
            X (data frame): the data frame that contains the input data on which the the learning will be performed
            Y (data frame): the data frame that contains the output data on which the the learning  will be performed
            category_names (list): the category names of the classifier
  
     """

    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('messages_cleaned', engine)
    X = df.message 
    Y = df.drop(columns=['message', 'original', 'genre'])
    category_names = list(Y.columns.values)
    return X, Y, category_names


def tokenize(text):
    """lowers and tokenizes the text
        
        Args:
            text (str): text to be tokenized

        Returns: 
            (list): a list of tokens from the original text


    """
    text = re.sub(r'[^0-9A-Za-z]'," ", text).lower()
    text = word_tokenize(text)
    return [w for w in text if not w in stop_words]


def build_model():
    """Function to build the model, the params are obtained by running cv.best_params_ 
       in the research notebook. 
       
       Note: The actual search for the best parameters is ommited as it took me about 3 days
       on a M1 apple computer to do the search

       Returns: 
             pipeline (pipeline): the pipeline with specified parameteres
    
    """
    
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize, max_df =  0.75, max_features = 5000, ngram_range = (1, 2))),
        ('tfidf', TfidfTransformer(use_idf =  True)),
        ('clf', MultiOutputClassifier(estimator=RandomForestClassifier(n_estimators = 200, min_samples_split = 2)))
    ])
    
    return pipeline
   
def evaluate_model(model, X_test, Y_test, category_names):
    """Evaluates the inputed model on the input data

    Args: 
        model (pipeline): the model that needs evaluation
        X_test (data frame): test input variable 
        Y_test (data frame): test output variable 
    Returns:
        classification_report (str): the classification report for the model
    
    """
    y_pred = model.predict(X_test)
    return classification_report(Y_test, y_pred, target_names = category_names)
    
    


def save_model(model, model_filepath):
    """Saves the model to the given filepath
        
        Args: 
            model (pipeline): model to be saved
            model_filepath (str): the filepath to where to save the model
    
    """
    joblib.dump(model, model_filepath)


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