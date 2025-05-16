import re 
import os

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.embeddings import HuggingFaceEmbeddings  # Updated import

# Ignoring warnings
import warnings
warnings.filterwarnings('ignore')

model_name = "sentence-transformers/all-MiniLM-L6-v2"
embedding_fn = HuggingFaceEmbeddings(model_name=model_name)

api_key = "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"

# Main model client
chat = ChatGroq(temperature=0, groq_api_key=api_key, model_name="llama-3.3-70b-versatile")

def document_loader():
    filename = "Dataset/customer_support_handbook.txt"
    with open(filename, errors="ignore") as file:
        contents = file.read()
    return contents

# Splitting the text by characters
print('Splitting the text by Char and tokens...')
char_splitter = RecursiveCharacterTextSplitter(
    separators=['\n', '\n\n', ' ', '. ', ', ', ''],
    chunk_size=1000,
    chunk_overlap=0.2
)

content = document_loader()
texts = char_splitter.create_documents([content])

vector_path = 'Vector_db/customer_support_handbook'
vector_db = FAISS.from_documents(documents=texts, embedding=embedding_fn)
vector_db.save_local(vector_path)

loaded_db = FAISS.load_local(
    vector_path,
    embeddings=embedding_fn,
    allow_dangerous_deserialization=True
)

# Create memory
memory = ConversationBufferWindowMemory(
    k=5,
    memory_key="chat_history",
    output_key="answer",
    return_messages=True,
)

# Create the ConversationalRetrievalChain
qa_conversation = ConversationalRetrievalChain.from_llm(
    llm=chat,
    chain_type="stuff",
    retriever=loaded_db.as_retriever(kwargs=15),
    return_source_documents=True,
    memory=memory
)

def QnA_Chain(query):
    print("Generating the response...")
    try:
        response = qa_conversation({"question": query})
        return response.get("answer", "I'm sorry, I didn't understand that.")
    except Exception as e:
        print(f"Error: {e}")
        return "An error occurred while generating the response."
