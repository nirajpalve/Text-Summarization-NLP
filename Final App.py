# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 09:31:47 2022

@author: Niraj Palve
"""

import numpy as np
import pandas as pd
import streamlit as st
from streamlit import caching 
from transformers import BartForConditionalGeneration, BartTokenizer, BartConfig
from summarizer import Summarizer
from sumy.parsers.plaintext import PlaintextParser

#App uses two NLP models to summarzie text. The larger and slower BART model is downloaded once the app runs and is cached. Caching will enable you to 
#test the trained BART model on as much tesxt as is needed without having to upload the model for each test. The Lex_Rank model is much smaller and faster.

# Loading the model and tokenizer for bart-large-cnn
@st.cache(allow_output_mutation=True)
def get_model():
    tokenizer=BartTokenizer.from_pretrained('facebook/bart-large-cnn')
    model1=BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
    return tokenizer,model1
#Run BART model 
tokenizer,model1= get_model() 

#Use pretrained model and tonekizer to produce summary tokens and then decode thise summarized tokens 
@st.cache(allow_output_mutation=True)
def summarizer(original_text):
    inputs = tokenizer.batch_encode_plus([original_text],return_tensors='pt')
    summary_ids=model1.generate(inputs['input_ids'],early_stopping=True)
    bart_summary=tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    #return BART summary
    return bart_summary

# Loading the model and tokenizer for bert model
@st.cache(allow_output_mutation=True)
def get_model():
    model = Summarizer()
    return model()
model=get_model
    

#Use pretrained BERT MODEL 
@st.cache(allow_output_mutation=True)
def bert_summarizer(original_text):
    model = Summarizer()
    bert_summary = model(original_text, min_length=20,max_length=500)
    return bert_summary


def main():
    """ NLP Based App with Streamlit. Choice of two mdodels. BART Large or BERT Model """

    # Title
    st.title("Text Summarizer")
    st.write("#")

     
    # Summarization    
    message = st.text_area("Enter Text in box below")
    st.write('#')

    option=st.selectbox('Select Summarization',('Bart Model','Bert Model'))
    if option ==  "Bart Model":
        summary_result = summarizer(message)
        st.write(summary_result)
    else :
            summary_result = bert_summarizer(message)
            st.write(summary_result)

if __name__== '__main__':
    main()


    