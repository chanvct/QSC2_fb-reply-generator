import os
import pandas as pd
import streamlit as st
from timeit import default_timer as timer


from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain import OpenAI
import openai

@st.cache_resource
def load_database():
    ### load data_info
    df = pd.read_csv('data_info_May23.csv')
    
    ### load vectordb
    persist_directory = './ChromaDB/'
    embedding = OpenAIEmbeddings(openai_api_key=st.secrets.openai.api_key)
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
    
    ### Semantic search
    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k":3})
    
    return df, vectordb, retriever

df, vectordb, retriever = load_database()


### Streamlit UI
header = st.container()
form = st.form(key='my_form')
response = st.container()
cite_sources = st.container()
footer = st.container()

with st.sidebar:
    st.markdown("**Description:**")
    st.markdown("This virtual assistant aims to help LTA officers draft a reply to a feedback/query sent in by the public.")
    st.markdown("It runs on OpenAI's GPT-3.5-Turbo model and has ingested information from the following sources: \
                LTA.gov.sg, MOT's parlimentary queries (Jan 2020 - Apr 2023), AskJamie (sample of 500), EFMS (sample of 100).\
                It will search the ingested information and use it in the reply ***if relevant**. \
                (Cited sources may turn out irrelevant due to lack of related content from LTA's website)")
    st.markdown("The LTA officer can also input comments to influence the reply based on the next steps, actions to be taken, decisions made, considerations etc.")
    st.markdown("~ App created by Valerie Chan | Data Science, ITCD")            
                
                
                

with header:
    st.title('QSC2 Feedback Reply Generator')
    
    
    
with form:
    user_input = st.text_area('**Input the feedback/query from the public:**')
    user_comments = st.text_input('**Comments from LTA officer (optional)**')
    submit_button = st.form_submit_button(label='Generate reply')
    
    
  
    if user_comments == '':  # if there are no comments
        
        template = """You are a smart virtual assistant from the Land Transport Authority (LTA) of Singapore. Your job is to help an LTA officer draft a reply to the following feedback or query (Feedback//Query) sent in by the public.
                    You can also make use of the knowledge in the source documents provided. Do not say anything that is not backed by facts from these documents. We are currently in April 2023.
                    
                    Feedback//Query: {user_input}
                    
                    Reply: """
        
        prompt_template = PromptTemplate(template=template, input_variables=['user_input'])
        prompt = prompt_template.format(user_input=user_input)
        
    else:   # if there are comments
    
        template = """You are a smart virtual assistant from the Land Transport Authority (LTA) of Singapore. Your job is to help an LTA officer draft a reply to the following feedback or query (Feedback//Query) sent in by the public.
                    The reply should incorporate the comments from the LTA officer below, which can consist of the next steps, actions to be taken, decisions made, considerations etc.
                    You can also make use of the knowledge in the source documents provided. Do not say anything that is not backed by facts from these documents. We are currently in April 2023.
                    
                    Feedback//Query: {user_input}
                    
                    Comments: {user_comments}
                    
                    Reply: """
        
        prompt_template = PromptTemplate(template=template, input_variables=['user_input','user_comments'])
        prompt = prompt_template.format(user_input=user_input, user_comments = user_comments)
        

if user_input == '':
    with response:        
        st.markdown('**Generated reply:**')
        st.write('<p style="font-size:14px">Please provide an input!</p>', unsafe_allow_html=True)


else:
    with response:
        
        st.markdown('**Generated reply by gpt-3.5-turbo:**')
        start = timer()
        qa = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(model_name = "gpt-3.5-turbo", max_tokens = 1000, temperature = 0.1, openai_api_key=st.secrets.openai.api_key), 
                        chain_type="stuff", 
                        retriever=retriever, 
                        return_source_documents=True)
        result = qa(prompt)
        st.write(result['result'])
        st.write("Time taken:", timer()-start)
        
              
    with cite_sources:
        st.markdown('--------------------------------------------------')
        st.markdown('**Sources:**')
        
        source_docs = retriever.get_relevant_documents(user_input)
        
        source_list = []
        for doc in source_docs:
            source = doc.metadata['source'].split('\\')[-1].strip('.txt')
            source_list.append(source)

        results_df = df[df['serial'].isin(source_list)]
        
        
        for index, row in results_df.iterrows():
            st.write(row['title'] + ' | ' + row['source'] + ' | ' + row['date'])
            
with footer:
    st.markdown('--------------------------------------------------')
    st.markdown('~ App created by Valerie Chan | Data Science, ITCD')