# -*- coding: utf-8 -*-
"""Streamlit Application for Gemma Model"""
pip install huggingface_hub

import os
from PIL import Image
import streamlit as st
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import keras
import keras_nlp

st.set_page_config(
    page_title="GenAI Magician",
    page_icon="✨",
    layout="centered",
    initial_sidebar_state="auto",
)

# Load GPT-2 model and tokenizer
@st.cache_resource
def load_gpt2_model():
    model_name = "openai-community/gpt2"
    model1 = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model1, tokenizer

# Load NousResearch/Llama-2-7b-chat-hf model and tokenizer
@st.cache_resource
def load_llama_nous_model():
    model_name = "NousResearch/Llama-2-7b-chat-hf"
    tokenizer2 = AutoTokenizer.from_pretrained(model_name)
    model2 = AutoModelForCausalLM.from_pretrained(model_name)
    return model2, tokenizer2

# Load Llama-2-7b-chat-finetune model and tokenizer
@st.cache_resource
def load_llama_finetune_model():
    model_name = "OumaimaABJAOU/Llama-2-7b-chat-finetune"
    model3 = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer3 = AutoTokenizer.from_pretrained(model_name)
    return model3, tokenizer3


# Load Gemma model and tokenizer
@st.cache_resource
def load_gemma_model():
    model_name = "google/gemma-2b-it"
    model4 = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer4 = AutoTokenizer.from_pretrained(model_name)
    return model4, tokenizer4

def generate_text_gpt2(prompt):
    inputs = gpt2_tokenizer.encode(prompt, return_tensors="pt")
    outputs = gpt2_model.generate(
        inputs, 
        max_length=150, 
        temperature=0.7, 
        num_return_sequences=1, 
        no_repeat_ngram_size=2,
    )
    text = gpt2_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text

def generate_text_llama1(prompt, model2, tokenizer2):
    inputs = tokenizer2.encode(prompt, return_tensors="pt")
    outputs = model2.generate(
        inputs, 
        max_length=150, 
        temperature=0.7, 
        num_return_sequences=1, 
        no_repeat_ngram_size=2,
    )
    text = tokenizer2.decode(outputs[0], skip_special_tokens=True)
    return text

def generate_text_llama2(prompt, model3, tokenizer3):
    inputs = tokenizer3.encode(prompt, return_tensors="pt")
    outputs = model3.generate(
        inputs, 
        max_length=150, 
        temperature=0.7, 
        num_return_sequences=1, 
        no_repeat_ngram_size=2,
    )
    text = tokenizer3.decode(outputs[0], skip_special_tokens=True)
    return text
def generate_text_gemma(prompt, model4, tokenizer4):
    inputs = tokenizer4.encode(prompt, return_tensors="pt")
    outputs = model4.generate(
        inputs, 
        max_length=150, 
        temperature=0.7, 
        num_return_sequences=1, 
        no_repeat_ngram_size=2,
    )
    text = tokenizer4.decode(outputs[0], skip_special_tokens=True)
    return text


# Load images
main_image = Image.open('C:/Users/hp/Desktop/Streamlit/static/main_banner.png')

# Set sidebar images and select box
format_type = st.sidebar.selectbox('Choose your GenAI magician 😉', ["GPT-2", "Llama-2-7b-chat-finetuned", "Llama-2-7b-chat-hf", "Gemma-2b-it"])

# Set the title dynamically based on the selected model
if format_type == "GPT-2":
    st.title("📄 GPT-2")
elif format_type == "Llama-2-7b-chat-finetuned":
    st.title("📄 Llama-2-7b-chat-finetuned")
elif format_type == "Llama-2-7b-chat-hf":
    st.title("📄 Llama-2-7b-chat-hf")
elif format_type == "Gemma-2b-it":
    st.title("📄 Gemma-2b-it")

# Load models
if format_type == "GPT-2":
    gpt2_model, gpt2_tokenizer = load_gpt2_model()
elif format_type == "Llama-2-7b-chat-finetune":
    llama_finetune_model, llama_finetune_tokenizer = load_llama_finetune_model()
elif format_type == "Llama-2-7b-chat-hf":
    llama_nous_model, llama_nous_tokenizer = load_llama_nous_model()
elif format_type == "Gemma-2b-it":
    gemma_model, gemma_tokenizer = load_gemma_model()

# Define behavior based on model selection
input_text = st.text_area("Please enter text here... 🙋", height=50)
chat_button = st.button("Do the Magic! ✨")

# Initialize generated_text
generated_text = ""

if chat_button and input_text.strip():
    with st.spinner("Loading...💫"):
        prompt = f"Answer the following question in a clear and concise manner:\n{input_text}\nAnswer:"
        if format_type == "GPT-2":
            generated_text = generate_text_gpt2(prompt)
        elif format_type == "Llama-2-7b-chat-finetune":
            generated_text = generate_text_llama2(prompt, llama_finetune_model, llama_finetune_tokenizer)
        elif format_type == "Llama-2-7b-chat-hf":
            generated_text = generate_text_llama1(prompt, llama_nous_model, llama_nous_tokenizer)
        elif format_type == "Gemma-2b-it":
            generated_text = generate_text_gemma(prompt, gemma_model, gemma_tokenizer)

        st.success(generated_text)
else:
    st.warning("Please enter something! ⚠")

# Add footer
st.markdown(
    "<br><hr><center>Made with ❤️ by <a href='mailto:abjaou.oumaima2002@gmail.com?subject=GenAI Magician WebApp!&body=Please specify the issue you are facing with the app.'><strong>Oumaimat</strong></a></center><hr>", 
    unsafe_allow_html=True
)
st.markdown("<style> footer {visibility: hidden;} </style>", unsafe_allow_html=True)
