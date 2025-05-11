import streamlit as st 
import os
import pickle
import pandas as pd
import re
from nltk.corpus import stopwords
    
script_dir = os.path.dirname(os.path.abspath(__file__))     
def load_model(file_path) :
    with open(file_path, 'r') as file :
        model_data = pickle.load(file)
    return model_data

stopword = set(stopwords.words('english'))
def preprocess_text(text):
    text = text.lower() 
    text = re.sub(r'\d+', '', text) 
    text = re.sub(r'\b\w{1,2}\b', '', text)  
    text = re.sub(r'[^\w\s]', '', text) 
    text = re.sub(r'\s+', ' ', text).strip() 
    text = ' '.join(word for word in text.split() if word not in stopword)
    return text

def load_content_model() :
    model_content = os.path.join(script_dir,'models','content.pkl')
    data_content = os.path.join(script_dir,'dataset','use','content_df.csv')
    
    with open(model_content, 'rb') as file :
        model_data = pickle.load(file)
        cos_sim = model_data['cos_sim']
        tfidf = model_data['tfidf']
    
    df = pd.read_csv(data_content)
    features = df[['Title', 'description', 'authors', 'publishedDate', 'categories', 'publisher']]
    features['publishedYear'] = features['publishedDate'].str[:4]
    features.drop(columns=['publishedDate'])
    features['content'] = features['description'] + ' ' + features['authors'] + ' ' + features['categories'] + 'publisher'
    features['clean_content'] = features['content'].apply(preprocess_text)
    features.set_index(features['Title'], inplace=True)
    features.index.name = 'Index Title'
    features.index = features.index.str.lower()
    
    return features, cos_sim

def load_collab_model():
    model_collab = os.path.join(script_dir,'models','collab.pkl')
    data_collab = os.path.join(script_dir,'dataset','use','collaborative_df.csv')
    
    with open(model_collab, 'rb') as file :
        model_data = pickle.load(file)
        user_sim_df = model_data['user_sim_df']
        user_books_matrix = model_data['user_books_matrix']
    
    df = pd.read_csv(data_collab)
    return df, user_books_matrix, user_sim_df
    
def recomendation(name, features, cos_sim, top_n=10) :
    idx = features.index.get_loc(name.lower())
    score = pd.Series(cos_sim[idx]).sort_values(ascending=False)
    top_score = list(score.iloc[1:top_n+1].index)
    recommended_books = features.iloc[top_score]['Title'].unique()
    return pd.DataFrame(recommended_books, columns=['Recommended Books'])
        
def user_recommendation(user_id, user_sim_df, user_books_matrix, df, top_n=10):
    sim_user = user_sim_df[user_id].sort_values(ascending=False).index[1:]
    sim_user_rating = user_books_matrix.loc[sim_user]
    user_no_rated = user_books_matrix.loc[user_id][user_books_matrix.loc[user_id]==0]
    recommendation = sim_user_rating[user_no_rated.index].mean().sort_values(ascending=False).head(top_n)
    recommended_books = df[df['Title'].isin(recommendation.index)]['Title'].unique()
    return pd.DataFrame(recommended_books, columns=['Recommended Books'])

def streamlit_app():
    st.title('Hybrid Filtering Books Recommended System')
    
    data_user = pd.read_csv('dataset/use/collaborative_df.csv')
    data_books = pd.read_csv('dataset/use/content_df.csv')

    features, cos_sim = load_content_model()
    df, user_books_matrix, user_sim_df = load_collab_model()
    
    user = st.selectbox('Choose User ID :', data_user['User_id'].unique())
    item = st.selectbox('Choose Desired Book :', data_books['Title'].unique())
    
    if st.button('Show Recommendation') :
        collab_recommendation = user_recommendation(user, user_sim_df, user_books_matrix, df, top_n=10)
        item_recommendation = recomendation(item, features, cos_sim, top_n=15)
        
        hybrid_recommendation = pd.concat([collab_recommendation, item_recommendation]).drop_duplicates(subset='Recommended Books').sample(frac=1, random_state=42).head(15)
        
        st.subheader('Books Based on Your Reading Activity')
        cols = st.columns(5)
        for i,title in enumerate(hybrid_recommendation['Recommended Books']) :
             if title in data_books['Title'].values :
                 books_info = data_books[data_books['Title'] == title]
                 book_image = books_info['image'].values[0]
                 try :
                     with cols[i % 5]:
                         st.image(book_image, width=200)
                 except Exception as e:
                         st.error(f"Error loading image for {title}: {e}")
                         
        st.markdown("""---""")                   
        st.subheader('Other Related Books')
        other_book = item_recommendation[~item_recommendation['Recommended Books'].isin(hybrid_recommendation['Recommended Books'])]
        cols = st.columns(5)
        for i,title in enumerate(other_book['Recommended Books']) :
             if title in data_books['Title'].values :
                 books_info = data_books[data_books['Title'] == title]
                 book_image = books_info['image'].values[0]
                 try :
                     with cols[i % 5]:
                         st.image(book_image, width=200)
                 except Exception as e:
                         st.error(f"Error loading image for {title}: {e}")
                         
if __name__=='__main__':
    streamlit_app()
