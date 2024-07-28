# Define Librarys
import streamlit as st
import nltk
from nltk.chat.util import Chat, reflections
import transformers
from deep_translator import GoogleTranslator
from warnings import filterwarnings
from transformers import AutoTokenizer,AutoModelWithLMHead,AutoModelForCausalLM
import torch
import nltk
nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
stemmer=LancasterStemmer()
import numpy as np
import random
import json
import pickle
from time import sleep
from tensorflow import keras
from keras import Sequential,layers

#--------------------------------------------------
#1-Selection part
with st.sidebar:
  st.title("Testing some Chatbots")
  st.subheader('Choose the model:')
  model = st.selectbox("Select", [ '1-Pairs','2-microsoft(arabic)','3-gpt_2','4-microsoft(English)'], key='pk1')
#-------------------------------------------------
#2-First Model
if model=='1-Pairs':
    st.sidebar.write("MAIN STEPS FOR THIS MODEL:")
    st.sidebar.write("1-Define conversation patterns and responses.")
    st.sidebar.write("2-Initialize the chatbot with these patterns.")
    st.sidebar.write("3-Create a Streamlit UI for user input and display the chatbot's responses.")
    pairs = [
                ["hello|hi|hey", ['Hello', 'Hi there!', 'Hey']],
                ['how are you|how is it going', ["I am fine, thanks", "I am doing well"]],
                ['what is your name', ["I'm a chatbot", "I am just a bot"]]
            ]
    chatbot = Chat(pairs=pairs, reflections=reflections)

    messages = st.container(height=300)
    if prompt := st.chat_input("Say something"):
      messages.chat_message("user").write(prompt)
      response = chatbot.respond(prompt.lower())

      messages.chat_message("assistant").write(f"chatbot: {response}")
#-----------------------------------------------------------
#3-Second Part(Microsoft)
if model=='2-microsoft(arabic)':
    st.sidebar.write("MAIN STEPS FOR THIS MODEL:")
    st.sidebar.write("1-Load the model(microsoft/DialoGPT-medium) and tokenizer from Hugging Face.")
    st.sidebar.write("2-Translate user input from Arabic to English.")
    st.sidebar.write("3-Generate a response using the model.")
    st.sidebar.write("4-Translate the response back to Arabic.")
    st.sidebar.write("5-Display the interaction in a Streamlit UI.")
    try:
        tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")  
        model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")  
        st.sidebar.write("Model and tokenizer loaded successfully")
        st.sidebar.write(f"Model type: {type(model)}")  
    except Exception as e:
        st.sidebar(f"Error loading model or tokenizer: {e}")

    # Initialize chat history
    chat_history_ids = None

    def chat(text): #arabic
        global chat_history_ids
        try:
            # Translate input text from Arabic to English
            text = GoogleTranslator(source='ar', target='en').translate(text)
            
            # Encode the user input
            new_user_input_ids = tokenizer.encode(text + tokenizer.eos_token, return_tensors='pt')
            
            # Concatenate new user input with chat history
            bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if chat_history_ids is not None else new_user_input_ids
            
            # Generate model response
            chat_history_ids = model.generate(
                bot_input_ids,
                max_length=1000,
                pad_token_id=tokenizer.eos_token_id,
                no_repeat_ngram_size=3,
                do_sample=True,
                top_k=100,
                top_p=0.7,
                temperature=0.8
            )
            
            # Decode the model response
            reply = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
            
            # Translate the response from English to Arabic
            text = GoogleTranslator(source='en', target='ar').translate(reply)
            
            return text
        except Exception as e:
            print(f"Error in chat function: {e}")
            return "Error occurred during chat processing"

    # Streamlit UI
    messages = st.container()
    if prompt := st.chat_input("Say something"):
        messages.chat_message("user").write(prompt)
        prompt = prompt.lower()
        response = chat(prompt)
        messages.chat_message("assistant").write(f"chatbot: {response}")

#-----------------------------------------------------------
#4-Third Model(GPT2)
if model=='3-gpt_2':
    st.sidebar.write("MAIN STEPS FOR THIS MODEL:")
    st.sidebar.write("1-Load the model(KnutJaegersberg/gpt2-chatbot) and tokenizer from Hugging Face.")
    st.sidebar.write("2-Generate a response using the model.")
    st.sidebar.write("3-Display the interaction in a Streamlit UI.")
    try:
        tokenizer = AutoTokenizer.from_pretrained("KnutJaegersberg/gpt2-chatbot")  # Replace "gpt2" with your model if different
        model = AutoModelForCausalLM.from_pretrained("KnutJaegersberg/gpt2-chatbot")  # Replace "gpt2" with your model if different
        st.sidebar.write("Model and tokenizer loaded successfully")
        st.sidebar.write(f"Model type: {type(model)}")  # Print the type of the model to confirm
    except Exception as e:
        st.sidebar(f"Error loading model or tokenizer: {e}")

    # Initialize chat history
    chat_history_ids = None

    def chat(text): #arabic
        global chat_history_ids
        try:
            # Translate input text from Arabic to English
            text = GoogleTranslator(source='ar', target='en').translate(text)
            
            # Encode the user input
            new_user_input_ids = tokenizer.encode(text + tokenizer.eos_token, return_tensors='pt')
            
            # Concatenate new user input with chat history
            bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if chat_history_ids is not None else new_user_input_ids
            
            # Generate model response
            chat_history_ids = model.generate(
                bot_input_ids,
                max_length=1000,
                pad_token_id=tokenizer.eos_token_id,
                no_repeat_ngram_size=3,
                do_sample=True,
                top_k=100,
                top_p=0.7,
                temperature=0.8
            )
            
            # Decode the model response
            reply = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
            
            # Translate the response from English to Arabic
            text = GoogleTranslator(source='en', target='ar').translate(reply)
            
            return text
        except Exception as e:
            print(f"Error in chat function: {e}")
            return "Error occurred during chat processing"

    # Streamlit UI
    messages = st.container()
    if prompt := st.chat_input("Say something"):
        messages.chat_message("user").write(prompt)
        prompt = prompt.lower()
        response = chat(prompt)
        messages.chat_message("assistant").write(f"chatbot: {response}")

#-----------------------------------------------------------
#5-fourth Part(Microsoft(arabic))
if model=='4-microsoft(English)':
    st.sidebar.write("MAIN STEPS FOR THIS MODEL:")
    st.sidebar.write("1-Load the model(microsoft/DialoGPT-medium) and tokenizer from Hugging Face.")
    st.sidebar.write("3-Generate a response using the model.")
    st.sidebar.write("5-Display the interaction in a Streamlit UI.")
    try:
        tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")  
        model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")  
        st.sidebar.write("Model and tokenizer loaded successfully")
        st.sidebar.write(f"Model type: {type(model)}")  
    except Exception as e:
        st.sidebar(f"Error loading model or tokenizer: {e}")

    # Initialize chat history
    chat_history_ids = None

    def chat(text): #arabic
        global chat_history_ids
        try:
            # Encode the user input
            new_user_input_ids = tokenizer.encode(text + tokenizer.eos_token, return_tensors='pt')
            
            # Concatenate new user input with chat history
            bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if chat_history_ids is not None else new_user_input_ids
            
            # Generate model response
            chat_history_ids = model.generate(
                bot_input_ids,
                max_length=1000,
                pad_token_id=tokenizer.eos_token_id,
                no_repeat_ngram_size=3,
                do_sample=True,
                top_k=100,
                top_p=0.7,
                temperature=0.8
            )
            
            # Decode the model response
            text = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
   
            return text
        except Exception as e:
            print(f"Error in chat function: {e}")
            return "Error occurred during chat processing"

    # Streamlit UI
    messages = st.container()
    if prompt := st.chat_input("Say something"):
        messages.chat_message("user").write(prompt)
        prompt = prompt.lower()
        response = chat(prompt)
        messages.chat_message("assistant").write(f"chatbot: {response}")

