from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate, MessagesPlaceholder

import streamlit as st
import streamlit_chat as message
import pyttsx3
from BlogDigAIUtils import *

openai_api_key = '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'

def text_2_voice(i) :
    print(f'text2voice and i value: {i}')
    voice_text = st.session_state['responses'][i]
    print(voice_text)
    try :
        pyttsx3.speak(voice_text)
    except Exception as e:
        print('Speak Button Error')

#st.title('Langchain Chatbot')
header_container = st.container()
with header_container :
    st.header('Dig Venkat Alagarsamy Blog', divider='rainbow')
#st.header('Dig Venkat Alagarsamy Blog', divider='rainbow')
#st.balloons()


user_html_string = '''
<style>
.bordered-container {
    /* border: 5px solid #007bff;
    padding: 10px; 
    border: 0px solid #d5d6d8; */
    border-radius: 5px;
    padding: 10px;
    background-color: #b9c5df
}
</style>

<div class="bordered-container">
    ___USER_QUERY___
</div>
'''

jarvis_html_string = '''
<style>
.bordered-container-jarvis {
    /* border: 5px solid #007bff;
    padding: 10px; 
    border: 0px solid #d5d6d8; */
    border-radius: 5px;
    padding: 10px;
    background-color: #d8dadc
}
</style>

<div class="bordered-container-jarvis">
    ___JARVIS_RESPONSE___
</div>
'''

if 'responses' not in st.session_state:
    st.session_state['responses'] = ["How can I assist you?"]

if 'requests' not in st.session_state:
    st.session_state['requests'] = []

if 'buffer_memory' not in st.session_state :
    st.session_state.buffer_memory=ConversationBufferWindowMemory(k=3, return_messages=True)

system_msg_template = SystemMessagePromptTemplate.from_template(template="""Answer the question as truthfully as possible using the provided context, and if the answer is not contained with the text below, say 'I don't know""" )

human_msg_template = HumanMessagePromptTemplate.from_template(template='{input}')

prompt_template = ChatPromptTemplate.from_messages([system_msg_template, MessagesPlaceholder(variable_name='history'), human_msg_template])

llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=openai_api_key)
conversation = ConversationChain(memory=st.session_state.buffer_memory, prompt=prompt_template, llm=llm, verbose=True)

response_container = st.container() #Chat History Container
text_container = st.container()     #User Input Textbox container


prompt = st.chat_input('Prompt: ', max_chars=1000)

with text_container :
    #prompt = st.chat_input('Prompt: ', max_chars=1000)
    if prompt:
        with st.spinner('Working...') :
            conversation_string = get_conversation_string()
            # st.code(conversation_string)
            refined_query = query_refiner(conversation_string, prompt)
            #st.subheader("Refined Query: ")
            #st.write(refined_query)
            context = find_match(refined_query)
            #print (f'context: {context}')
            predict_input = f'Context:\n {context} \n\n Query:\n {prompt}'
            #print (f'Predict Input: {predict_input}')
            response = conversation.predict(input=predict_input)


        st.session_state.requests.append(prompt)
        st.session_state.responses.append(response) 

with response_container :
    if st.session_state['responses']:

        for i in range(len(st.session_state['responses'])):

            #col1, col2 = st.columns([0.8,0.2])
            with st.columns([0.9,0.1])[0]: 
            #with col1:
                with st.chat_message('assistant') :
                    #st.write(st.session_state['responses'][i])
                    jarvis_html = jarvis_html_string.replace('___JARVIS_RESPONSE___', st.session_state['responses'][i])
                    col1, col2 = st.columns([0.9,0.1])
                    with col1:
                        st.markdown(jarvis_html, unsafe_allow_html=True)
                    with col2:
                        st.button('S', on_click=text_2_voice, args=[i], key=f'button_{i}')

    
        
            #with col2:
                #st.button('Speak', on_click=text_2_voice)

            with st.columns([0.2,0.8])[1]:
            #with col2:
                with st.chat_message('user') :
                    #st.write('Venkat icon')
                    if i < len(st.session_state['requests'] ):
                        #print (f"{i} value is {st.session_state['requests'][i]}")
                        #st.write(st.session_state['requests'][i])
                        user_html = user_html_string.replace('___USER_QUERY___', st.session_state['requests'][i])
                        st.markdown(user_html, unsafe_allow_html=True)

       
                    
                        

                
