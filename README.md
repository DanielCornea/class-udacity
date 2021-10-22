# Udacity-Project-classification

## Instalation 
This project requires: 
   Python 3: packages: skllearn, nltk, pandas, flask, json, plotly


## Project overview
The aim of this project is to clasify messages according to their category. 
The source of the messages is supposed to be twitter, the messages are supposed to 
be received during a natural disaster event.

## File Descriptions 
APP: 
--TEMPLATES:
---go.html and master.html contain the presentation html code for the browser
--run.py contains the flask app and the code used to retrieve and present the data
DATA:
--- .csv files contain the input data used for the ML algorithm 
--- DisasterResponse.db is the database used to store data 
--- process_data.py contains the code to clean and save the data to the DB 
MODELS: 
--- train_classifier.py contains the ML algo code


## Running Instructions

In order to run the code from the source folder of this project run: 

python app/run.py


## Licensing Authors and Acknowledgements
This project is authored by me, Daniel Cornea, as part of the Udacity's Data Scientist nanodegree