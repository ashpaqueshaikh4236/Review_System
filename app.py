import requests
from bs4 import BeautifulSoup
import nltk
import streamlit as st
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('stopwords')
ps = PorterStemmer()


def fetch_reviews(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        
        reviews = []
        review_elements = soup.find_all('div', class_='text show-more__control')
        
        for review_element in review_elements:
            review_text = review_element.get_text().strip()
            reviews.append(review_text)
        
        return reviews
    
    else:
        st.error(f"Failed to retrieve page, status code: {response.status_code}")
        return None


def stemming(content):
    if not content.strip():
        return ''
    
    stemmed_content = re.sub('[^a-zA-Z]',' ',content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [ps.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)

    return stemmed_content


tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("Review checking System")

url = st.text_input('Enter url')

if st.button('Submit'):
    if not url.strip() :
        st.warning('Please enter url')
    else:
        # Fetch Data
        reviews = fetch_reviews(url)

        for index, review in enumerate(reviews, start=1):
            # 1. Preprocess
            transformed_sms = stemming(review)

            # 2. vectorize
            vector_input = tfidf.transform([transformed_sms])

            # 3.predict
            result = model.predict(vector_input)[0]

            # 4. Display
            if result == 1:
                st.write(f"Review {index}: {review}")
                st.success("Postive")
            else:
                st.write(f"Review {index}: {review}")
                st.warning("Negative")


























# import streamlit as st
# import requests
# from bs4 import BeautifulSoup
# import nltk
# import pickle

# def fetch_reviews(url):
#     response = requests.get(url)
#     soup = BeautifulSoup(response.text, 'html.parser')
#     reviews = []
#     for review_tag in soup.find_all('div', class_='review-text'):
#         reviews.append(review_tag.get_text())
    
#     return reviews

# tfidf = pickle.load(open('vectorizer.pkl','rb'))
# model = pickle.load(open('model.pkl','rb'))


# st.title("Amazon Review checking")

# url = st.text_input('Enter url')


# if st.button('Predict'):
#     if not url.strip() :
#         st.warning('Please Write Something!')
#     else:
#         # 1. Preprocess
#         transformed_sms = stemming(input_sms)

#         # 2. vectorize
#         vector_input = tfidf.transform([transformed_sms])

#         # 3.predict
#         result = model.predict(vector_input)[0]

#         # 4. Display
#         if result == 1:
#             st.success("Postive")
#         else:
#             st.warning("Negative")