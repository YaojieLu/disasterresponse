import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    This function loads the content of messages and categories datafiles into a Pandas Dataframe.
    Input:
    - messages_filepath(String): location of messages.csv file
    - categories_filepath(String): location of categories.csv file
    Output:
    - df(Dataframe): messages and categories tables merged on 'id'
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories,on='id')
    return df


def clean_data(df):
    """
    This function cleans the Dataframe containing messages and categories, 
    performs some transformation to get it ready for ML pipeline.
    Input:
    - df(Dataframe): the Dataframe containing messages and categories
    Output:
    - df(Dataframe): the cleaned and transformed Dataframe from input
    """
    categories = df['categories'].str.split(';',expand=True)
    row = categories.iloc[0,:].copy()
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
    
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    df = df.drop(['categories'],axis=1)
    df = pd.concat([df, categories], axis = 1)
    df = df[~df.duplicated()]
    df.related.replace(2,1,inplace=True)
    return df     

def save_data(df, database_filename):
    """
    This function saves the cleaned Dataframe 'df' into a SQLite database file. 
    Input:
    - df(Dataframe): the cleaned and transformed Dataframe containing messages and categories
    - database_filename(String): location to store the database file
    """
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('df', engine, index=False)
    return


def main():
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