import os
os.environ['http_proxy'] = "http://proxy.h2.de:8080" 
os.environ['https_proxy'] = "http://proxy.h2.de:8080" 

import streamlit as st
from llama_index.core import VectorStoreIndex, Document
from llama_index.core import Settings
from llama_index.llms.openai_like import OpenAILike
#from OpenAILike import OpenAILike
import openai
from llama_index.core import SimpleDirectoryReader


# Lokale Definitionen
api_base = "https://ai.h2.de/llm/v1"
api_key = "your_api_key"
model_name = "Llama-3.1-SauerkrautLM-70b-Instruct"
openai.api_key = api_key

# Initialisiere Sprachmodell
Settings.llm = OpenAILike(api_base=api_base, api_key=api_key, model=model_name, max_tokens=256, temperature=0.8, system_prompt="Du bist Experte für den Studiengang AI.Engineering. Das Gespräch soll sich um die Studienordnung und den Studiengang drehen. Deine Antworten entsprechen den Fakten entsprechend der verfügbaren Dokumente - halluziniere keine Fakten.")

### Initialisiere Embeddings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings

Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)


# Initialisiere den Chat:
st.header("Fragen zum Studiengang AI.Engineering 💬")

if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "Ich beantworte dir Fragen zur Studienordnung des Studiengangs AI.Engineering an der Otto-von-Guericke Universität Magdeburg sowie an den Hochschulen Anhalt, Harz, Magdeburg-Stendal und Merseburg"}
    ]

# Dokumente indexieren
@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Ich lese die abgelegten Dokumente. Moment..."):
        reader = SimpleDirectoryReader(input_dir="./chatdata", recursive=True)
        docs = reader.load_data()
        
        index = VectorStoreIndex.from_documents(docs, show_progress=True)
        return index

index = load_data()


# Chat-Objekt erstellen
chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)

# Gespräch starten
if prompt := st.chat_input("Deine Fragen:"): # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])


# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Denke nach..."):
            response = chat_engine.chat(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message) # Add response to message history