#!/usr/bin/env python
# coding: utf-8

# # deployment

# In[12]:


import streamlit as st


# In[13]:


from PIL import Image
image = Image.open('ApplAiOnly_Logo.png')
st.image(image)
with st.form(key="form1"):
    st.title("Text Summarizer")
    text=st.text_input(label="Enter the required text")
    submit=st.form_submit_button(label="Abstractive Summary")
    submit2=st.form_submit_button(label="Extractive Summary")
    


# # 1)Abstractive Summary

# # Installing the required libraries

# In[5]:


 !pip install transformers


# In[6]:


 !pip install torch==1.4.0


# # importing the needed libraries

# In[4]:


import torch
import json 
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config


# # Setting the model

# In[5]:


model = T5ForConditionalGeneration.from_pretrained('t5-small')
tokenizer = T5Tokenizer.from_pretrained('t5-small')
device = torch.device('cpu')


# # summarizing the text

# In[6]:


def Asummarize(text):
    preprocess_text = text.strip().replace("\n","")
    t5_prepared_Text = "summarize: "+preprocess_text
    tokenized_text = tokenizer.encode(t5_prepared_Text, return_tensors="pt").to(device)
    # summmarize 
    summary_ids = model.generate(tokenized_text,
                                        num_beams=10,
                                        no_repeat_ngram_size=2,
                                        min_length=100,
                                        max_length=200,
                                        early_stopping=True)

    output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return output


# # 2)Extractive Summary

# # Installing the required libraries

# In[1]:


from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import nltk
nltk.download('stopwords')
nltk.download('punkt')


# # summarizing the text

# In[2]:


def Esummarize(text):
    # Tokenizing the text
    stopWords = set(stopwords.words("english"))
    words = word_tokenize(text)

    # Creating a frequency table to keep the
    # score of each word

    freqTable = dict()
    for word in words:
        word = word.lower()
        if word in stopWords:
            continue
        if word in freqTable:
            freqTable[word] += 1
        else:
            freqTable[word] = 1

    # Creating a dictionary to keep the score
    # of each sentence
    sentences = sent_tokenize(text)
    sentenceValue = dict()

    for sentence in sentences:
        for word, freq in freqTable.items():
            if word in sentence.lower():
                if sentence in sentenceValue:
                    sentenceValue[sentence] += freq
                else:
                    sentenceValue[sentence] = freq



    sumValues = 0
    for sentence in sentenceValue:
        sumValues += sentenceValue[sentence]

    # Average value of a sentence from the original text

    average = int(sumValues / len(sentenceValue))

    # Storing sentences into our summary.
    summary = ''
    for sentence in sentences:
        if (sentence in sentenceValue) and (sentenceValue[sentence] > (0.8 * average)):
            summary += " " + sentence
    return summary


# In[10]:


if(submit==True):
    outputt=Asummarize(text)
    st.subheader(outputt, anchor=None)
    submit=False
if(submit2==True):
    outputt2=Esummarize(text)
    st.subheader(outputt2, anchor=None)
    submit2=False

