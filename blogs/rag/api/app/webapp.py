import json
import streamlit as st
from typing import Callable
from langchain.chains import ConversationChain
from langchain.llms.sagemaker_endpoint import SagemakerEndpoint
from langchain.llms.sagemaker_endpoint import ContentHandlerBase

# Sagemaker Endpoint
SAGEMAKER_ENDPOINT_FOR_TEXT2TEXT_GENERATION = "qa-w-rag-huggingface-text2text-flan-t5--2023-04-15-13-25-17-476"
AWS_REGION = "us-east-1"

# Set Streamlit page configuration
st.set_page_config(page_title='🧠MemoryBot🤖', layout='wide')
# Initialize session states
if "generated" not in st.session_state:
    st.session_state["generated"] = []
if "past" not in st.session_state:
    st.session_state["past"] = []
if "input" not in st.session_state:
    st.session_state["input"] = ""
if "stored_session" not in st.session_state:
    st.session_state["stored_session"] = []


def setup_sagemaker_endpoint_for_text_generation(endpoint_name: str=SAGEMAKER_ENDPOINT_FOR_TEXT2TEXT_GENERATION, region: str = AWS_REGION) -> Callable:
    parameters = {
    "max_length": 200,
    "num_return_sequences": 1,
    "top_k": 250,
    "top_p": 0.95,
    "do_sample": False,
    "temperature": 1,}

    content_handler = ContentHandlerForTextGeneration()    
    
    sm_llm = SagemakerEndpoint(
        endpoint_name=endpoint_name,
        region_name=region,
        model_kwargs=parameters,
        content_handler=content_handler)
    return sm_llm

class ContentHandlerForTextGeneration(ContentHandlerBase):
    content_type = "application/json"
    accepts = "application/json"
    
    def transform_input(self, prompt: str, model_kwargs = {}) -> bytes:
        input_str = json.dumps({"text_inputs": prompt, **model_kwargs})
        return input_str.encode('utf-8')
    
    def transform_output(self, output: bytes) -> str:
        response_json = json.loads(output.read().decode("utf-8"))
        return response_json["generated_texts"][0]

# Define function to get user input
def get_text():
    """
    Get the user input text.

    Returns:
        (str): The text entered by the user
    """
    input_text = st.text_input("You: ", st.session_state["input"], key="input",
                            placeholder="I am an LLM powered assistant! Ask me something ...", 
                            label_visibility='hidden')
    return input_text


# Set up sidebar with various options
with st.sidebar.expander("🛠️ ", expanded=False):
    TEXT2TEXT_MODEL = st.selectbox(label='Text2Text Model', options=['flan-t5-xxl'])
    EMBEDDING_MODEL = st.selectbox(label='Embeddings Model', options=['gpt-j-6b'])
    

# Set up the Streamlit app layout
st.title("🤖 Chat Bot")
st.subheader(" Powered by 🦜 LangChain + Amazon Sagemaker + Streamlit")

llm = setup_sagemaker_endpoint_for_text_generation()

   
# Create the ConversationChain object with the specified configuration
Conversation = ConversationChain(
        llm=llm
    )  

# Get the user input
user_input = get_text()

# Generate the output using the ConversationChain object and the user input, and add the input/output to the session
if user_input:
    print(f"user_input={user_input}")
    output = Conversation.run(input=user_input)  
    st.session_state.past.append(user_input)  
    st.session_state.generated.append(output)  

# Allow to download as well
download_str = []
# Display the conversation history using an expander, and allow the user to download it
with st.expander("Conversation", expanded=True):
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        st.info(st.session_state["past"][i],icon="🧐")
        st.success(st.session_state["generated"][i], icon="🤖")
        download_str.append(st.session_state["past"][i])
        download_str.append(st.session_state["generated"][i])
    
    # Can throw error - requires fix
    download_str = '\n'.join(download_str)
    if download_str:
        st.download_button('Download',download_str)

# Display stored conversation sessions in the sidebar
for i, sublist in enumerate(st.session_state.stored_session):
        with st.sidebar.expander(label= f"Conversation-Session:{i}"):
            st.write(sublist)

# Allow the user to clear all stored conversation sessions
if st.session_state.stored_session:   
    if st.sidebar.checkbox("Clear-all"):
        del st.session_state.stored_session
