from matplotlib import image
import streamlit as st
import pickle
from keybert import KeyBERT
import numpy as np
from newsapi import NewsApiClient
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

st.set_page_config(page_title="Fake News Detector", page_icon=":newspaper:", layout='wide')

st.title("Fake News Detector")
st.subheader(":dart: Check how credible a news source is :dart:")


st.sidebar.header(":gear:News API Parameters")
st.sidebar.info("Adjust the parameters below to get the results you want.")
st.sidebar.write("---")
start_date = st.sidebar.date_input("Select Start Date")
lang = st.sidebar.selectbox("Select a language", ["en", "fr", "sp"], index=0)
page_size = st.sidebar.slider("News Api Search Result Size", 20, 100, value = 50)

st.sidebar.write("---")

st.sidebar.header(":dart: How it Works")
st.sidebar.write("""
                The fake news detection is based on the concept of stance detection.

                * Users input a claim like "Elon Musk Acquires Twitter!"
                * The program will search different global and local news sources for their 'stance' on that topic (claim).
                * The sources are then passed through our Reputability Algorithm. If lots of reputable sources all agree with the claim, then it's probably true!
                * Then we cite our sources so users can click through and read more about that topic!
                """)


# Load css file
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ---- LOAD FILES ----
@st.cache(allow_output_mutation=True)
def load(model_name):
    model = load_model(model_name)
    key_bert_model = KeyBERT()
    headline_tokenizer = pickle.load(open("headline_tokenizer.pickle", "rb"))
    body_tokenizer = pickle.load(open("body_tokenizer.pickle", "rb"))

    return model, key_bert_model, headline_tokenizer, body_tokenizer

model, key_bert_model, headline_tokenizer, body_tokenizer = load('perfect_model')

# ---- HELPER FUNTIONS ----
def predict(head, body, model):
    encoded_docs_headline = headline_tokenizer.texts_to_sequences([head]) 
    padded_docs_headline = pad_sequences(encoded_docs_headline, 15, padding='post', truncating='post')
    
    encoded_docs_body = body_tokenizer.texts_to_sequences([body]) 
    padded_docs_body = pad_sequences(encoded_docs_body, 40, padding='post', truncating='post')
  
    res = model.predict([padded_docs_headline, padded_docs_body])

    stance = {0:"Agree",
              1:"Disagree",
              2:"Discuss",
              3:"Unrelated"}
    
    return stance[np.argmax(res)]


def search(model, claim=None):

    Agree = 0
    Disagree = 0
    Unrelated = 0
            
    isReal = False
    isFake = False

    data = []

    try:
        if claim != "":
            key_words = key_bert_model.extract_keywords(claim, top_n=1, \
                keyphrase_ngram_range=(1,5), stop_words='english')
            key_words = key_words[0][0]

            newsapi = NewsApiClient(api_key='a5bdaefa54ef4ddcacecfc76d8747434')
            result = newsapi.get_everything(q=key_words, page_size=page_size, \
                                language=lang, from_param=start_date)

            articles = result['articles']
        
            for _, article in enumerate(articles):
                stance = predict(claim, article['description'], model)

                data.append({"Title":article['title'],
                            "Source": article['source']["name"],
                            "Decription":article['description'],
                            "Stance":stance,
                            "Link":article['url'],
                            #"Content":article['content'],
                            })
                
                if stance == "Agree":
                    Agree += 1
                elif stance == 'Disagree':
                    Disagree += 1
                elif stance == 'Unrelated':
                    Unrelated += 1
        else:
            pass
        
    except ConnectionError as e:
        print(e)

    if Agree > Disagree:
        isReal = True
        
    elif Disagree > Agree:
        isFake = True
    
    elif Unrelated > Disagree:
        isFake = True
    
    return isReal, isFake, data


with st.container():
    st.write("---")
    claim = st.text_input("Enter a claim (headline)", "")
    st.write("#")

# Run the search function
Real, Fake, Data = search(model, claim)

if Real == True:
    st.markdown("Based on the sources we checked and referenced this article against, it's probably *credible!*:smiley:")
    # st.image("real.jpg", use_column_width=True)
    st.success("Real News!")
    # st.markdown(f"<h1 style='color:green; text-align:center'>Real News!</h1>", unsafe_allow_html=True)
  
elif Fake == True:
    st.image("fake.jpg", use_column_width=True)
    st.markdown("Whoops! Based on the sources we checked and referenced this article against, it's probably *not credible!*:warning:")
    # st.error("Fake News!")
else:
    st.empty()

with st.container():
    st.write("---")
    st.write("Here are some articles that were found from other reputable news sources!")
    st.table(Data)