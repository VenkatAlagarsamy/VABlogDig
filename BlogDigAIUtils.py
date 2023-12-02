#from sentence_transformers import SentenceTransformer
from langchain.vectorstores import qdrant
from qdrant_client import QdrantClient
from langchain.chains import RetrievalQA
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
import os
#import pyttsx3
import pyaudio
#import pinecone
from openai import OpenAI
import streamlit as st

openai_api_key = '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'
#openai.api_key = openai_api_key
#API Keys
OPENAI_API_KEY = openai_api_key
QDRANT_API_KEY = 'XBO7p356R4TQ4DGkgkI48Nx2rbUXlWTP8NOdx8hfAq_OkOvnT9qe1Q'
QDRANT_END_POINT = 'https://27994267-5aae-4840-b101-85dba72dfcd8.us-east4-0.gcp.cloud.qdrant.io:6333'
my_collection_name = 'venkat-blog-collection'
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

openai_client = OpenAI(api_key='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')

"""
t2v_engine = pyttsx3.init()
voices = t2v_engine.getProperty('voices')
t2v_engine.setProperty('voice',voices[1].id)
t2v_engine.setProperty('rate',150)
t2v_engine.runAndWait()
print ('calling utils')
#t2v_engine.say('Please feel free to dig Venkat Alagarsamy Blog')
#t2v_engine.runAndWait()

model = SentenceTransformer('all-MiniLM-L6-v2')
pinecone.init(api_key='', environment='us-east-1-aws')
index = pinecone.Index('langchain-chatbot')
"""

#print('Create Qdrant Client')
client = QdrantClient (
    url = QDRANT_END_POINT,
    api_key = QDRANT_API_KEY
)

def find_match(query):
    """
    input_em = model.encode(input).tolist()
    result = index.query(input_em, top_k=2, includeMetadata=True)
    return result['matches'][0]['metadata']['text']+"\n"+result['matches'][1]['metadata']['text']

    
    embeddings = OpenAIEmbeddings()
    #embeddings = HuggingFaceInstructEmbeddings(model_name='hkunlp/instructor-xl')

    vectorstore = qdrant.Qdrant(
        client=client, 
        collection_name=my_collection_name, 
        embeddings=embeddings,
    )
    
    emodel = ChatOpenAI(api_key=openai_api_key)
    try :
        qa = RetrievalQA.from_chain_type (
            llm = emodel,
            chain_type = 'stuff',
            retriever = vectorstore.as_retriever()
        )

        response = qa.run(query)

        return (response)
    
    except Exception as e:
        print(f'qa exception: {e}')
        return (None)
    """
    return ('find match return')


def query_refiner(conversation, query):
    """
    #prompt=f'Given the following user query and conversation log, formulate a question that would be the most relevant to provide the user with an answer from a knowledge base.\n\nCONVERSATION LOG: \n{conversation}\n\nQuery: {query}\n\nRefined Query:'
    #print(f'@query_refiner - prompt: {prompt}')
    
    response = openai_client.completions.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0.7,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    #return response['choices'][0]['text']
    return response.choices[0].text
    """
    return ('response from query refiner')

def get_conversation_string():
    conversation_string = ""
    for i in range(len(st.session_state['responses'])-1):
        
        conversation_string += "Human: "+st.session_state['requests'][i] + "\n"
        conversation_string += "Bot: "+ st.session_state['responses'][i+1] + "\n"
    return conversation_string


"""
def text_2_voice(i) :
    print(f'text2voice and i value: {i}')
    voice_text = st.session_state['responses'][i]
    print(voice_text)

    if t2v_engine.isBusy():
        t2v_engine.endLoop()
   
    t2v_engine.say(voice_text)
    t2v_engine.runAndWait()
    print('run completed')
"""
