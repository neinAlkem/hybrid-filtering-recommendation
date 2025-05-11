import pandas as pd
import os

def cleaning_data(books='dataset/raw/books_data.csv', user='dataset/raw/Books_rating.csv') :
  
  """
  The function `cleaning_data` processes and cleans data from two datasets, merges them, filters out
  popular books and active users, and saves the cleaned data into new CSV files.
  
  :param books: The `books` parameter in the `cleaning_data` function is used to specify the file path
  of the dataset containing information about books. By default, it is set to
  'dataset/raw/books_data.csv', defaults to dataset/raw/books_data.csv (optional)
  :param user: The `user` parameter in the `cleaning_data` function is the file path to the dataset
  containing user ratings for books. In this case, it is set to 'dataset/raw/Books_rating.csv'. This
  dataset is used to merge with the books dataset to clean and process the data for further, defaults
  to dataset/raw/Books_rating.csv (optional)
  """
  
  print('Load original data..')
  content_data = pd.read_csv(books, header=0)
  collaborative_data = pd.read_csv(user, header=0)
  
  print('Processing data..')
  books_info = pd.merge(content_data, collaborative_data, on='Title', how='inner')
  books_info.drop_duplicates(inplace=True)
  books_info.dropna(inplace=True)
  popular_books = books_info['Title'].value_counts().head(1000).index
  active_user = books_info['User_id'].value_counts().head(15000).index
  content_df = content_data[content_data['Title'].isin(popular_books)]
  user_df = collaborative_data[collaborative_data['User_id'].isin(active_user)]
  user_content_df = user_df[user_df['Title'].isin(popular_books)]
  
  df_cleaned = user_content_df[(user_content_df['review/score'] <= 5)]
  user_review_count = df_cleaned['User_id'].value_counts()
  active_users = user_review_count[user_review_count > 10].index
  df_cleaned = df_cleaned[df_cleaned['User_id'].isin(active_users)]
  
  print('Saving..')
  if not os.path.exists('dataset/use') :
    os.makedirs('dataset/use', exist_ok=True)
    
  df_cleaned.to_csv('dataset/use/collaborative_df.csv', index=False)
  content_df.to_csv('dataset/use/content_df.csv', index=False)

cleaning_data()


