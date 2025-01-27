import sys


def load_data(messages_filepath, categories_filepath):
        """
    Loads data
    Args:
    messages_filepath: The path of the messages dataset
    categories_filepath: The path of the categories dataset
    """
    # import libraries
    import pandas as pd
    import numpy as np
    from sqlalchemy import create_engine
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    # load categories dataset
    categories = pd.read_csv(categories_filepath)


def clean_data(df):
    """
    Cleanse data
    Args:
    df: the loaded dataset
    """
    # merge datasets
    df = pd.merge(messages, categories, on='id', how='left')
    categories = categories['categories'].str.split(';', expand=True)
    # select the first row of the categories dataframe
    row = categories[:1]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row.applymap(lambda x: x[:-2]).iloc[0, :].tolist()
    print(category_colnames)
    # rename the columns of `categories`
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1]

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
      # drop the original categories column from `df`
    categories.drop('child_alone', axis = 1, inplace = True)
    df.drop('categories', axis = 1, inplace = True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories],axis=1)
    # drop duplicates and NaNs:
    df.drop_duplicates(inplace=True)
    df = df[df['related'] != 2]
    df =df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]



def save_data(df, database_filename):
    """
    Saves the data
    Args:
    df: the loaded dataset
    database_filename: The path of the database file
    """
    engine = create_engine('sqlite:///disaster.db')
    df.to_sql(database_filename, engine, index=False, if_exists='replace')  


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
