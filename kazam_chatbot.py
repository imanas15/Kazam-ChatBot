from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
import json
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferWindowMemory  # No longer needed in this case
from langchain_community.embeddings import HuggingFaceEmbeddings  # Updated import

# Ignoring warnings
import warnings
warnings.filterwarnings('ignore')

# Define the model and embedding
model_name = "sentence-transformers/all-MiniLM-L6-v2"
embedding_fn = HuggingFaceEmbeddings(model_name=model_name)

api_key = "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"

# Main model client (Groq-based LLM)
chat = ChatGroq(temperature=0, groq_api_key=api_key, model_name="mixtral-8x7b-32768")

# Function to load document contents
def document_loader():
    filename = "Dataset/complete_data_sb.txt"
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

# Load and split content
content = document_loader()
texts = char_splitter.create_documents([content])

# Vector store for FAISS (storing the document vectors)
vector_path = 'Vector_db/complete_data_sb'
vector_db = FAISS.from_documents(documents=texts, embedding=embedding_fn)
vector_db.save_local(vector_path)

# Load the FAISS vector store
loaded_db = FAISS.load_local(
    vector_path,
    embeddings=embedding_fn,
    allow_dangerous_deserialization=True
)

# Create the RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=chat,
    chain_type="stuff",  # Using the 'stuff' chain type, can be swapped with 'map_reduce' or 'refine'
    retriever=loaded_db.as_retriever(kwargs=15)  # FAISS retriever with a threshold of 15
)

# Function to handle QnA queries
def QnA_Chain(queries):
    print("Generating the response...")
    try:
        responses = {}
        for query in queries:
            # Adding instruction for concise answers (1 or 2 words only)
            modified_query = f"Please answer with only one word: {query}"
            
            # Get the response for the modified query
            response = qa_chain({"query": modified_query})
            
            # Store the response in the dictionary
            responses[query] = response.get("result", "I'm sorry, I didn't understand that.")
        
        # Return responses as JSON-like dictionary
        return json.dumps(responses, indent=4)
    except Exception as e:
        print(f"Error: {e}")
        return "An error occurred while generating the response."

# Example usage
queries = [
    "what is the cost for Therapeutic radiology?",
    "what is the cost for Maximum out-of-pocket amount?",
    "what is the cost for lab services?",
    "what is the cost for Diagnostic  radiology services (e.g. MRI, CT  1,2  scan)?",
    "what is the cost for Diagnostic tests and procedures?",
    "what is the cost for Routine eyewear in vision services?",
    "what is the cost for Outpatient group therapy visit?",
]

print(QnA_Chain(queries))