import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import HumanMessage, SystemMessage
from gtts import gTTS
import tempfile
from pypdf import PdfReader
import pandas as pd                   
import os
from dotenv import load_dotenv
load_dotenv()

os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")

# Language selection
languages = {
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Chinese": "zh",
    "Japanese": "ja",
    "Hindi": "hi",
    "Arabic": "ar"
}

llm = ChatOpenAI(model="gpt-3.5-turbo")

# Core functionalities
def translate(input: str, lang: str) -> str:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size= 4000, chunk_overlap=200)
    chunks = text_splitter.split_text(input)
    trans_chunk = []
    for chunk in chunks:
        messages = [
            SystemMessage(content=f"You are an expert translator. Translate the following text to {lang}."),
            HumanMessage(content=chunk),
        ]
        response = llm.invoke(messages)
        trans_chunk.append(response.content)
        
    return " ".join(trans_chunk)

def text_to_speech(text, code):
    tts = gTTS(text, lang= code)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3", dir='D:\Gen_AI\Edureka_pro\Audio_Files') as tmpfile:
        tts.save(tmpfile.name)
        return tmpfile.name
    
    
def extract_text_from_uploads(doc):
    if doc.name.endswith(".pdf"):
        reader = PdfReader(doc)
        text = ""
        for i in range(len(reader.pages)):
            page = reader.pages[i]
            if page:
                text = text + page.extract_text() +" "
        return text.strip() if text else "No text could be extracted from the File."
    
    elif doc.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file, encoding="utf-8")
        return df.to_string(index=False)
    
    elif doc.name.endswith(".xlsx") or doc.name.endswith(".xls"):
        df = pd.read_excel(uploaded_file)
        return df.to_string(index= False)
    
    elif doc.name.endswith(".txt"):
        return uploaded_file.read().decode("utf-8")
    
    else:
        return "Unsupported file type."

# UI
st.title("Language Translator + Text-to-Speech App")
user_input = st.radio("Choose input method:", ("Type Text", "Upload File"))

if user_input == "Type Text":
    text = st.text_area("Enter text to translate:", height=200)
elif user_input == "Upload File":
    uploaded_file = st.file_uploader("Choose a file", type= ["pdf","txt","xlsx","xls","csv"])
    if uploaded_file is not None:
        text = extract_text_from_uploads(uploaded_file)
        st.text_area("Extracted text:", text,  height=200)
language = st.radio("Choose language into which you want to translate: ", list(languages.keys()))     
lang_code = languages[language]

if st.button("Translate"):
    if not text:
        st.warning("Please enter a valid text or upload a file in pdf, txt, excel or csv format.")
    else:
        translated_text = translate(text, language)
        st.success("Translated Text")
        st.write(translated_text)
        
        with st.spinner("Converting to speech...."):
            dir_path = text_to_speech(text, lang_code)
            st.audio(dir_path, format="audio/mp3")
            # To download the file
            with open(dir_path, "rb") as file:
                st.download_button( label="Download Audio File", data=file, file_name="translated_speech.mp3", icon=":material/download:")
                
        
        