import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
nltk.download('wordnet')

from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///{}'.format('../data/DisasterResponse.db'))
print(engine)
df = pd.read_sql_table('messages_cleaned', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # second graph 
    
    df_cp = df.drop(columns=['message', 'genre', 'original'])
    print(df.head())
    df_cp = df_cp.sum().sort_values(ascending=False)

    category_names = list(df_cp.index)
    category_counts = list(df_cp.values)

    # third graph

    df_cp2 = df.copy()
    df_cp2.drop(columns=["message", "genre"], inplace=True)
    df_cp2["original_length"] = df_cp2.original.str.len()
    df_cp2.drop(columns=['original'], inplace = True)
    ls = []
    for col in df_cp2.columns:
        avg = df_cp2.query('{} == 1'.format(col))["original_length"].mean()
        ls.append((col, avg))

    df_three = pd.DataFrame(ls, columns =["cat", "avg"])
    df_three.dropna(inplace = True)
    df_three.drop(df_three[df_three.cat == "original_length"].index, inplace=True)
    
    avg_names = list(df_three["cat"])
    avg_counts = list(df_three["avg"])
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        # second graph
        {
            'data': [
                Bar(
                    y=category_counts,
                    x=category_names, 
                    # orientation = 'h', 
                    # yaxis_title_standoff = "30"
                    # title_standoff =  200,
                 
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories',
                  'yaxis': {
                   
                    'title': "count"
                },
                'xaxis': {
                      'title': {
                            'text': "Temprature",
                            'standoff': '30'
                     }
                }
            }
        }, 

         # third graph
        {
            'data': [
                Bar(
                    x=avg_names,
                    y=avg_counts
                )
            ],

            'layout': {
                'title': 'Average Length of messages by category',
                'yaxis': {
                    'automargin': "true",
                    'title': {
                            'text': "Average",
                            'standoff': "40"
                            }
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()