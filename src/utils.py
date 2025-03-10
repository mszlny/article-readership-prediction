import pandas as pd
import re
import joblib

def preprocess_article(article_title, lead_article_category):
    '''
    Preprocesses an article title and prepares it for model prediction.
    
    Parameters:
    - article_title (str): The raw title of the article.
    - lead_article_category (str): The category of the article ('Afternoon', 'Early morning', etc.).
    
    Returns:
    - pd.DataFrame: A processed DataFrame ready for prediction.
    '''
    
    # Create DataFrame with title and lead article category features
    article_df = pd.DataFrame({
        'Title': [article_title],
        'Lead article?_Afternoon lead article': [1 if lead_article_category == 'Afternoon' else 0],
        'Lead article?_Early morning lead article': [1 if lead_article_category == 'Early morning' else 0],
        'Lead article?_Morning lead article': [1 if lead_article_category == 'Morning' else 0],
        'Lead article?_Not lead article': [1 if lead_article_category == 'Not lead' else 0],
        'Lead article?_Tabloid lead article': [1 if lead_article_category == 'Tabloid' else 0]
    })

    #Adding stop words
    from nltk.corpus import stopwords
    #Adding unique stop words
    unique_stop_words = ['is', 'ha', 'le', 'fog', '2024', '2024ben', 'te', 'előtt', 'dolog', 'nap', 'párizs', 'kvíz', 'mire', 'olimpia']
    stop_words = set(stopwords.words('hungarian'))
    stop_words.update(unique_stop_words)
    
    # Preprocess the title
    cleaned_title = article_df['Title'].str.lower()  # Convert to lowercase
    cleaned_title = cleaned_title.apply(lambda x: re.sub(r'[^a-záéíóöőúüű0-9\s]', '', x))  # Remove special characters
    cleaned_title = cleaned_title.str.replace(r'\s+', ' ', regex=True).str.strip()  # Normalize spaces
    cleaned_title = cleaned_title.apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))  # Remove stopwords
    
    # Load the TF-IDF vectorizer (this is not uploaded to GitHub!)
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    
    # Transform the cleaned title into numerical features
    title_vectorized = vectorizer.transform(cleaned_title).toarray()
    
    # Convert vectorized data into DataFrame and merge with lead article features
    title_vectorized_df = pd.DataFrame(title_vectorized)
    processed_article = pd.concat([title_vectorized_df, article_df.drop(columns=['Title'])], axis=1)
    
    # Convert column names to strings (required for RandomForestRegressor)
    processed_article.columns = processed_article.columns.astype(str)
    
    return processed_article
