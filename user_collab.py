import pandas as pd
import numpy as np
import os
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

def user_collaboration(data_path='dataset/use/collaborative_df.csv', model_save_path='models/content.pkl'):
    
    """
    The function `user_collaboration` builds a user-based collaborative filtering model for recommending
    books based on user similarities.
    
    :param data_path: The `data_path` parameter in the `user_collaboration` function is the file path to
    the CSV file containing the user collaborative data. This data is read into a DataFrame to perform
    collaborative filtering for recommending books to users based on their similarities with other
    users, defaults to dataset/use/collaborative_df.csv (optional)
    :param model_save_path: The `model_save_path` parameter in the `user_collaboration` function is the
    file path where the trained model will be saved as a pickle file. This model includes the user
    similarity matrix (`user_sim_df`) and the user-books matrix (`user_books_matrix`), defaults to
    models/content.pkl (optional)
    :return: The `user_collaboration` function returns the `user_sim_df` and `user_books_matrix` after
    saving the model in the specified `model_save_path`.
    """
    
    df = pd.read_csv(data_path, header=0)

    features = df[['User_id', 'Title', 'review/score']]
    features.drop_duplicates(subset=['User_id', 'Title'], inplace=True)

    user_books_matrix = features.pivot(index='User_id', columns='Title', values='review/score')
    user_books_matrix.fillna(0, inplace=True)

    scaler = StandardScaler()
    normalized_matrix = scaler.fit_transform(user_books_matrix)

    user_sim = cosine_similarity(normalized_matrix)
    np.fill_diagonal(user_sim, 0)
    user_sim_df = pd.DataFrame(user_sim, index=user_books_matrix.index, columns=user_books_matrix.index)
    
    if not os.path.exists(model_save_path) :
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        
    with open(model_save_path, 'wb') as file :
        pickle.dump({'user_sim_df' : user_sim_df, 'user_books_matrix' : user_books_matrix}, file)
    
    def user_recommendation(user_id, user_sim_df, user_books_matrix, df, top_n=10):
            sim_user = user_sim_df[user_id].sort_values(ascending=False).index[1:]
            sim_user_rating = user_books_matrix.loc[sim_user]

            user_no_rated = user_books_matrix.loc[user_id.upper()][user_books_matrix.loc[user_id.upper()]==0]
            recommendation = sim_user_rating[user_no_rated.index].mean().sort_values(ascending=False).head(top_n)
            recommended_books = df[df['Title'].isin(recommendation.index)]['Title'].unique()
            return pd.DataFrame(recommended_books, columns=['Recommended Books'])

    print(f'Model saved succesfully in : {model_save_path}')
    return user_sim_df, user_books_matrix



