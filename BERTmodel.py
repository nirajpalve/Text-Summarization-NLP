# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 19:51:49 2022

@author: Niraj N Palve
"""

import pandas as pd
import streamlit as st
from streamlit import caching 
from summarizer import Summarizer

# Loading the model and tokenizer for bert
@st.cache(allow_output_mutation=True)
def get_model():
    model = Summarizer()
    return model()
model=get_model
	

#Use pretrained BERT MODEL 
@st.cache(allow_output_mutation=True)
def summarizer(original_text):
    model = Summarizer()
    bert_summary = model(original_text, min_length=20,max_length=500)
    return bert_summary

def main():
	""" NLP Based App with Streamlit. Choice of two mdodels. BART Large or Sumy Lex Rank """

	# Title
	st.title("Text Summarizer (BERT Model)")
	st.write("#")

	 
	# Summarization	
	message = st.text_area("Enter Text in box below")
	st.write('#')

	#provide choice of model. 
	if st.button("Summarize"):			 
		summary_result = summarizer(message)	
		st.success(summary_result)

if __name__ == '__main__':
	main()
