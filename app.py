import getpass
import os
from langchain_groq.chat_models import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate

# Fetch the API key from environment variables instead of hardcoding it
# In Colab, you can set this using userdata.get('groq'), but in a local or GitHub environment,
# you'll need to set the environment variable manually or load it from a .env file.

# For Colab users (Google Colab environment):
# from google.colab import userdata
# key = userdata.get('groq')
# os.environ["GROQ_API_KEY"] = key

# For non-Colab users (local machine or GitHub environment):
# It's recommended to set the API key manually in your environment or use a .env file:
# Example for local or GitHub use:
# os.environ["GROQ_API_KEY"] = "your-api-key-here"

# Alternatively, you can load environment variables from a .env file using dotenv (recommended for local setups):
# from dotenv import load_dotenv
# load_dotenv()  # This will load environment variables from a .env file
# os.environ["GROQ_API_KEY"] = os.getenv('GROQ_API_KEY')

# Initialize LLM (example)
from langchain_groq.chat_models import ChatGroq
llm = ChatGroq(
    model_name="deepseek-r1-distill-llama-70b",
    temperature=0.2
)

# Initialize LLM
llm = ChatGroq(
    model_name="deepseek-r1-distill-llama-70b",
    temperature=0.2
)

# Load the document
def load_documents(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents

# Split text
def split_the_text(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = splitter.split_documents(docs)
    return texts

# Use Chroma vectorstore
def vectorstore_from_documents(texts):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma.from_documents(texts, embedding=embeddings)
    return db

# Define chain
def conversational_chain(vectorstore):
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key='answer')

    prompt_template = """
    You are an advanced AI assistant capable of answering questions based on the provided context.
    Follow a structured step-by-step reasoning process to enhance accuracy.
    
    **Guidelines:**
    - **Do not hallucinate information.** If the answer is not in the context, respond with: "I'm sorry, but the answer is not available in the provided documents."
    - **Extract relevant details** from the context and summarize concisely.
    - **Ensure logical consistency** by verifying information before answering.

    **Thought Process:**
    1. Understand the key intent of the question.
    2. Retrieve the most relevant information from the context.
    3. Validate the retrieved details against prior responses.
    4. Construct a well-reasoned, structured, and concise response.
    
    **Context:**
    {context}

    **Chat History:**
    {chat_history}

    **User Question:**
    {question}


    **Final Answer:**
    """

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "chat_history", "question"]
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt},
        return_source_documents=True,
        verbose=True
    )
    return chain

# Main
def main(file_path):
    docs = load_documents(file_path)
    if not docs:
        raise ValueError("No documents were loaded. Check the file path or PDF content.")

    texts = split_the_text(docs)
    if not texts:
        raise ValueError("No text chunks were generated. Check the PDF content or text splitter config.")

    vectorstore = vectorstore_from_documents(texts)
    chain = conversational_chain(vectorstore)
    return chain

#Example usage
file_path = "path_to_your_pdf_document.pdf"   
chain = main(file_path)

#  Interact
response = chain.invoke({"question": "Give me the Summary of document?"})
print(response["answer"])
