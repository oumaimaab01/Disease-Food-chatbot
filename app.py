import os
from dotenv import load_dotenv
from PIL import Image
import streamlit as st
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer
from gradientai import Gradient

st.set_page_config(
    page_title="GenAI Magician",
    page_icon="✨",
    layout="centered",
    initial_sidebar_state="auto",
)


# Charger les variables d'environnement depuis le fichier .env
load_dotenv()

# Accéder aux variables d'environnement
gradient_access_token = os.getenv('GRADIENT_ACCESS_TOKEN')
gradient_workspace_id = os.getenv('GRADIENT_WORKSPACE_ID')

print(f"Gradient Access Token: {gradient_access_token}")
print(f"Gradient Workspace ID: {gradient_workspace_id}")

st.write("Enter your Hugging Face token to authenticate.")
token = st.text_input("Hugging Face Token", type="password")

if token:
    try:
        login(token)
        st.success("Successfully logged in to Hugging Face")
    except Exception as e:
        st.error(f"Login failed: {e}")
        st.stop()
        

# Load GPT-2 model and tokenizer
@st.cache_resource
def load_gpt2_model():
    model_name = "openai-community/gpt2"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

# Load Gemma model and tokenizer
@st.cache_resource
def load_gemma_model():
    model_name = "google/gemma-2b-it"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def generate_text_gpt2(prompt, model, tokenizer):
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(
        inputs, 
        max_length=150, 
        temperature=0.7, 
        num_return_sequences=1, 
        no_repeat_ngram_size=2,
    )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text

def generate_text_gemma(prompt, model, tokenizer):
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(
        inputs, 
        max_length=150, 
        num_return_sequences=1, 
        no_repeat_ngram_size=2,
        do_sample=True
    )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text

def get_response_from_llama2(prompt):
    try:
        with Gradient() as gradient:
            model_adapter = gradient.get_base_model(base_model_slug="llama2-7b-chat")
            completion = model_adapter.complete(query=prompt, max_generated_token_count=500).generated_output
            return completion
    except Exception as e:
        return f"Erreur avec le modèle Llama2: {e}"

# Load images
main_image = Image.open('static/main_banner.png')

# Set sidebar images and select box
format_type = st.sidebar.selectbox('Choose your GenAI magician 😉', ["GPT-2", "Gemma-2b-it", "llama2-7b-chat"])


# Set main image and title
st.image(main_image, use_column_width='auto')

# Set the title dynamically based on the selected model
if format_type == "GPT-2":
    st.title("📄 GPT-2")
elif format_type == "Gemma-2b-it":
    st.title("📄 Gemma-2b-it")
elif format_type == "llama2-7b-chat":
    st.title("📄 llama2-7b-chat")

# Load models
if format_type == "GPT-2":
    model, tokenizer = load_gpt2_model()
elif format_type == "Gemma-2b-it":
    model, tokenizer = load_gemma_model()

# Define behavior based on model selection
input_text = st.text_area("Please enter text here... 🙋", height=50)
chat_button = st.button("Do the Magic! ✨")

# Initialize generated_text
generated_text = ""


if chat_button and input_text.strip():
    with st.spinner("Loading...💫"):
        prompt = f"Answer the following question in a clear and concise manner:\n{input_text}\nAnswer:"
        if format_type == "GPT-2":
            generated_text = generate_text_gpt2(prompt, model, tokenizer)
        elif format_type == "Gemma-2b-it":
            generated_text = generate_text_gemma(prompt, model, tokenizer)
        elif format_type == "llama2-7b-chat":
            generated_text = get_response_from_llama2(prompt)

        st.success(generated_text)

else:
    st.warning("Please enter something! ⚠")

# Add footer
st.markdown(
    "<br><hr><center>Made with ❤️ by <a href='mailto:abjaou.oumaima2002@gmail.com?subject=GenAI Magician WebApp!&body=Please specify the issue you are facing with the app.'><strong>Oumaimat</strong></a></center><hr>", 
    unsafe_allow_html=True
)
st.markdown("<style> footer {visibility: hidden;} </style>", unsafe_allow_html=True)
