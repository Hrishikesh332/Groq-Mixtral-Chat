import streamlit as st
import os
import random
from langchain.chains import ConversationChain
from groq import Groq
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
 

load_dotenv()

groq_api_key = os.environ['API_KEY']

def main():
    st.markdown("<h1 style='text-align: center;';>Groq Chat - Mixtral 👨🏻‍💻</h1>", unsafe_allow_html=True)
    st.markdown("---")
    st.sidebar.title('Select an LLM')
    model = 'mixtral-8x7b-32768'
    conversational_memory_length = st.sidebar.slider('Conversational memory length:', 1, 10, value = 5)

    memory=ConversationBufferWindowMemory(k=conversational_memory_length)

    user_question = st.text_area("Ask a question:")
    s=st.button("submit")

    if s:
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history=[]
        else:
            for message in st.session_state.chat_history:
                memory.save_context({'input':message['human']},{'output':message['AI']})

        groq_chat = ChatGroq(
                groq_api_key=groq_api_key, 
                model_name=model
        )

        conversation = ConversationChain(
                llm=groq_chat,
                memory=memory
        )

        if user_question:
            response = conversation(user_question)
            message = {'human':user_question,'AI':response['response']}
            st.session_state.chat_history.append(message)
            st.write("Chatbot:", response['response'])

if __name__ == "__main__":
    main()
