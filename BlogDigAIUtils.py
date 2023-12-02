#Import Lagnchain packages
from langchain.vectorstores import qdrant           #Vector DB to manage high-dimensional vectors
from langchain.chains import RetrievalQA            #Q&A system to answer with high accuracy and efficiency
from langchain.embeddings import OpenAIEmbeddings   #To access OpenAI pre-trained text embedding nmodels
#from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.chat_models import ChatOpenAI        # ChatOpenAI to communicate with OpenAI GPT-3 model

#Import othe packages such as QdrantClient, OpenAI, pyttsx3 and streamli
from qdrant_client import QdrantClient              #Easy-to-use interface to interact(append,insert,search,delete) with Qdrant Vector Engine and manage collections and vector metadata
from openai import OpenAI   #Provides access to OpenAI models to generate text, translate language, write creative content and Q&A
import pyttsx3              #Text-to-speech (TTS) library
import streamlit as st      #To create and share custom web apps using Python
import os                   #To set environment varaibles

#import custom (BlogDig) specific packages
from global_vars import *

#set API keys, urls and environments
os.environ['OPENAI_API_KEY'] = get_openai_api_key()
openai_api_key = get_openai_api_key()
qdrant_api_key = get_qdrant_api_key()
qdrant_url  = get_qdrant_url()
qdrant_collection_name = get_qdrant_collection_name()

#define openai client to interact with OpenAI models and engines
openai_client = OpenAI(api_key=openai_api_key)

#define qdrant client to interact with Qdrant Vector Database
qdrant_client = QdrantClient (
    url = qdrant_url,
    api_key = qdrant_api_key
)

#function get the entire conversation from session state
def get_conversation_string():
    conversation_string = ""
    for i in range(len(st.session_state['responses'])-1):
        conversation_string += "Human: "+st.session_state['requests'][i] + "\n"
        conversation_string += "Bot: "+ st.session_state['responses'][i+1] + "\n"
    return conversation_string

#Fucntion to fine tune user prompt 
def get_refined_query(conversation, prompt):

    #rewrite the user prompt with the better prompt for instruction to model
    fine_tuned_prompt=f'Given the following user query and conversation log, formulate a question that would be the most relevant to provide the user with an answer from a knowledge base.\n\nCONVERSATION LOG: \n{conversation}\n\nQuery: {prompt}\n\nRefined Query:'
    #print(f'@query_refiner - prompt: {prompt}')

    #generate text completions using OpenAI pre-trained language model
    response = openai_client.completions.create(
        model='text-davinci-003',           #OpenAI Language model
        prompt=fine_tuned_prompt,           #Finetuned Prompt
        temperature=0.7,                    #High Creativity
        max_tokens=256,                     #Max words
        top_p=1,                            #Most likely completions. highr number is set to get more diversed response(range 0.0 to 1.0)
        frequency_penalty=0,                #Allows any number of repetion of words (only if it fits the context)
        presence_penalty=0                  #No control over instructiong to generate certain keywords
    )

    return response.choices[0].text

#Functrion to get context from refined(fine-tunned) query(prompt)
def get_chat_context(refined_query):

    qdrant_embeddings = OpenAIEmbeddings() #paid embedding model
    #embeddings = HuggingFaceInstructEmbeddings(model_name='hkunlp/instructor-xl')  #alternate opensource embedding model

    #create vectorstore by interacting through qdrant_client and embeddings using OpenAI model
    #Knowldge source collction is already created and stored in qdrant vector db using another program
    try :
        vectorstore = qdrant.Qdrant(
            client=qdrant_client,               
            collection_name=qdrant_collection_name, 
            embeddings=qdrant_embeddings            # the choice of embeddings model to be used 
        )
    except Exception as e:
        print (f'Vector store creation Exception:{e}')

    llm = ChatOpenAI(api_key=openai_api_key)    #LLM is defined using ChatOpenAI
    try :
        #creates qa instance to use agumented transformers to answer questions (combined and organized before passign to transformer model)
        qa = RetrievalQA.from_chain_type (
            llm = llm,
            chain_type = 'stuff', #In Langchain, 'stuff' argument denotes a default approach to concatenate retrived documents in single chain
            retriever = vectorstore.as_retriever()
        )
        #Generate response within a Q&A model.I/P: RefinedQuery O/P: IntelligentResponse Proceeser: LLM like ChatOpenAI()
        context = qa.run(refined_query) 
        return (context)
    except Exception as e:
        print(f'RetrievalQA exception: {e}') #Most of the time it throws error due to unavailability of llm engine or invalid subscription
        return (None)

#Function to convert text to voice and speak
#Converts the user requested speak (button) - bot response where i is the response request number
def text_2_voice(i) :
    #print(f'text2voice and i value: {i}')
    voice_text = st.session_state['responses'][i]
    #print(voice_text)
    try :
        pyttsx3.speak(voice_text)
    except Exception as e:
        print('Speak Button Error')  #This exception is added to ignore the exception when user clicks speak button continously

