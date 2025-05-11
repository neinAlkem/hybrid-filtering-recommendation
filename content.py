
import pandas as pd
import os
import math as math
import re
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import pickle

def build_content_model(data_path='dataset/use/content_df.csv', model_save_path='models/content.pkl'):
    
    """
    The function `build_content_model` preprocesses text data, calculates cosine similarity, and saves
    the model for recommending books based on content.
    
    :param data_path: The `data_path` parameter in the `build_content_model` function is the file path
    to the CSV file containing the content data. By default, it is set to 'dataset/use/content_df.csv'.
    This CSV file likely contains information about titles, descriptions, authors, published dates,
    categories, and, defaults to dataset/use/content_df.csv (optional)
    :param model_save_path: The `model_save_path` parameter in the `build_content_model` function is the
    file path where the trained content model will be saved as a pickle file. This model file will
    contain the cosine similarity matrix (`cos_sim`) and the TF-IDF vectorizer (`tf`) used for
    content-based recommendations, defaults to models/content.pkl (optional)
    :return: The function `build_content_model` returns the following three objects:
    1. `features`: DataFrame containing the processed features extracted from the input data.
    2. `cos_sim`: The cosine similarity matrix calculated based on the TF-IDF vectors of the content.
    3. `tf`: The TF-IDF vectorizer object used to transform the content data.
    """
    
    df = pd.read_csv(data_path)
    features = df[['Title', 'description', 'authors', 'publishedDate', 'categories', 'publisher']]
    features['publishedYear'] = features['publishedDate'].str[:4]
    features.drop(columns=['publishedDate'])
    features['content'] = features['description'] + ' ' + features['authors'] + ' ' + features['categories'] + 'publisher'

    stopword = set(stopwords.words('english'))
    def preprocess_text(text):
        text = text.lower() 
        text = re.sub(r'\d+', '', text) 
        text = re.sub(r'\b\w{1,2}\b', '', text)  
        text = re.sub(r'[^\w\s]', '', text) 
        text = re.sub(r'\s+', ' ', text).strip() 
        text = ' '.join(word for word in text.split() if word not in stopword)
        return text

    features['clean_content'] = features['content'].apply(preprocess_text)
    features.set_index(features['Title'], inplace=True)

    tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), stop_words='english')
    tfidf = tf.fit_transform(features['clean_content'])
    cos_sim = cosine_similarity(tfidf, tfidf)

    if not os.path.exists('model_save_path') :
        os.makedirs(os.path.dirname('model_save_path'), exist_ok=True)

    with open(model_save_path, 'wb') as file :
        pickle.dump({'cos_sim' : cos_sim, 'tfidf' : tf}, file)
        
    def recomendation(name,cos_sim=cos_sim, top_n=10) :
        idx = features.index.get_loc(name)
        score = pd.Series(cos_sim[idx]).sort_values(ascending=False)
        top_score = list(score.iloc[1:top_n+1].index)

        recommended_books = features.iloc[top_score]['Title'].unique()

        return pd.DataFrame(recommended_books, columns=['Recommended Books'])

    print(f'Model saved succesfully in : {model_save_path}')
    return features, cos_sim, tf
