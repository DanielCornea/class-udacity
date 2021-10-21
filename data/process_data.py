import sys
from sqlalchemy import create_engine
import pandas as pd 


def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv(messages_filepath, index_col='id')
    categories =  pd.read_csv(categories_filepath,  sep="[,]", engine='python', index_col='id')
    return messages.merge(categories, on = 'id')
    

def clean_data(df):
    """Function to clean the data
       
       Args: 
           df (dataframe) - previously loaded dataframe 

       Returns:
           df (dataframe) - cleaned dataframe    
    """
    # splitting categories  
    categories = df.categories.str.split(pat = ";", expand = True) 
    categories.columns = [column.split("-")[0] for column in categories.iloc[0]]
    

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1:]
    
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
        
    df.drop(columns = ['categories'], inplace=True)
    df = df.merge(categories, on = 'id')
    
    # droping duplicates
    df.drop_duplicates(subset=['original'], inplace=True)

    # eliminating value '2' from the columns    
    for column in df.columns:
        if 2 in df[column].unique():
            df.drop(df.index[df[column] == 2], inplace = True)
    
   
    return df


def save_data(df, database_filename):
    """Saving the data to database

       Args:
           df (dataframe) - previously cleaned data
               
    """
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('messages_cleaned', engine, index=False, if_exists='replace')
      


def main():
    """Self explanatory: the main function
    """
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)
        

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
      
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()