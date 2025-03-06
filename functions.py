 # Import Langchain modules
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from pydantic import Field
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings

# Other modules and packages
import streamlit as st  
import pandas as pd
import uuid
import re
import os
import io
import tempfile

# Clean filename function
def clean_filename(filename):
    """
    Remove "(number)" pattern from a filename 
    (because this could cause error when used as collection name when creating Chroma database).
    """
    # Regular expression to find "(number)" pattern
    new_filename = re.sub(r'\s\(\d+\)', '', filename)
    
    return new_filename

#Get PDF text 
def get_pdf_text(uploaded_file): 

    #create a temporary file to avoid empty list error
    temp_file = None

    try:
        # Read file content
        input_file = uploaded_file.read()

        # Check if file is empty
        if not input_file:
            raise ValueError("Cannot read an empty file")

        # Create a temporary file (PyPDFLoader requires a file path to read the PDF,
        # it can't work directly with file-like objects or byte streams that we get from Streamlit's uploaded_file)
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(input_file)
        temp_file.close()

        # Load PDF document
        loader = PyPDFLoader(temp_file.name)
        documents = loader.load()

        return documents
    
    except Exception as e:
        # Catch any error, print it for debugging purposes
        st.error(f"An error occurred while processing the PDF: {str(e)}")
        return []

    finally:
        # Ensure the temporary file is deleted when we're done with it
        if temp_file and os.path.exists(temp_file.name):
            os.unlink(temp_file.name)

#split the text from the PDF document into chuncks
def split_document(documents, chunk_size, chunk_overlap):    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                                  chunk_overlap=chunk_overlap,
                                                  length_function=len,
                                                  separators=["\n\n", "\n", " "])
    
    # Split the documents into chunks
    chunks = text_splitter.split_documents(documents)
    
    # Ensure we only keep valid chunks (non-empty and non-None)
    chunks = [chunk for chunk in chunks if chunk.page_content and chunk.page_content.strip()]
    
    return chunks

#Get embeddings for the vectorstore
def get_embedding_function():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return embeddings

#Vectorstore to hold the text chunks and embeddings
def create_vectorstore(chunks, embedding_function, file_name, vector_store_path="db"):
    """
    Create a vector store from a list of text chunks.
    """
    # Filter out any chunks with None or empty content
    valid_chunks = [chunk for chunk in chunks if chunk.page_content and chunk.page_content.strip()]
    
    # Ensure that we only process valid chunks
    if not valid_chunks:
        raise ValueError(f"No valid content to process in {file_name}. Skipping.")
    
    # Process valid chunks (this is where the embedding happens)
    ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, doc.page_content)) for doc in valid_chunks]
    unique_chunks = list({uuid: chunk for uuid, chunk in zip(ids, valid_chunks)}.values())

    # Create the vector store
    vectorstore = Chroma.from_documents(documents=unique_chunks, 
                                        collection_name=clean_filename(file_name),
                                        embedding=embedding_function, 
                                        ids=ids, 
                                        persist_directory=vector_store_path)
    
    vectorstore.persist()
    
    return vectorstore

#Further refine the vecotorstore
def create_vectorstore_from_texts(documents, file_name):
    """
    Create a vector store from a list of texts.
    """

    # Split the documents into chunks
    chunks = split_document(documents, chunk_size=1000, chunk_overlap=200)
    
    # Step 3 define embedding function
    embedding_function = get_embedding_function()

    # Step 4 create a vector store  
    vectorstore = create_vectorstore(chunks, embedding_function, file_name)
    
    return vectorstore

# Build the prompt template for the question-answering task
PROMPT_TEMPLATE = """
You are an assistant chatbot from the FDA, here to help novice users understand important FDA guidance on cosmetics.
Use the following pieces of retrieved context from the context of the uploaded files to answer the question.
If the information is unclear or not found in the context, you should acknowledge that you don't know the answer.
For forward-looking questions, you can speculate based on your understanding, but ensure your speculation is
grounded in current trends, scientific advancements, and historical context. If there is a direct answer to the question,
do not speculate DO NOT make up details.

Please format your answer as a clear, concise response and, when applicable, include sample statements or illustrative examples to make the explanation easier to understand. Here's how you can approach it:
- Start by summarizing the key points from the context (if available).
- Provide an example or a concrete statement where possible.
- When speculating, consider technological advances, changes in consumer behavior, and emerging trends. Provide concrete predictions or examples of what these products might look like based on these factors.
- End with a direct answer to the user's question, based on the context provided.

Context:
{context}

---

Question: {question}

Answer:
"""

"""Retrieval Pipeline"""
def contextualize_query(query):
    # This function modifies the original query by adding context
    # For general document questions, we prepend with a request to summarize
    return f"Provide a summary of FDA guidelines in this document: {query}"

def generate_answer_from_context(context, query):
    """
    Generate an answer based on the retrieved context from PDFs.
    """
    # Build a prompt template with the context and the user's question
    prompt_template = ChatPromptTemplate.from_template("""
        Context: {context}

        Question: {query}

        Answer:
    """)
    
    # Pass the context and the question to the prompt template and use an LLM to generate the answer
    llm = ChatOllama(model='llama3.2', temperature=0.8)
    
    # Construct the chain for generating the response
    rag_chain = prompt_template | llm

       # Assuming the response is returned as a generator
    response_text = ''
    for chunk in response_stream:
        response_text += chunk  # Incrementally build the response text
        yield response_text  # Yield each chunk of the response as it comes
    
    # Generate the answer by invoking the chain
    response_stream = rag_chain.invoke({"context": context, "query": query})

    return response_text

#Formats documents into a string
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

#Query the vector store with a question and return a structured response.
def query_document(vectorstore, query="Extract relevant details from the uploaded file."):
 
    llm = ChatOllama(model='llama3.2', temperature=0)
    retriever = vectorstore.as_retriever(search_type="similarity")
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    # Check if the query is valid
    if not query or not isinstance(query, str):
        raise ValueError(f"Invalid query: {query}")
    
    # Prepare the retrieval chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt_template
        | llm
    )

    # Execute the chain
    structured_response = rag_chain.invoke(query)

    return structured_response
#Process multiple PDF files, extract relevant data, and return the results as a DataFrame.
def process_multiple_pdfs(pdf_files, query):

    results = []

    for pdf_file in pdf_files:
        file_name = pdf_file.name
        documents = get_pdf_text(pdf_file)

        if not documents:
            st.warning(f"No content found in {file_name}. Skipping.")
            continue

        # Check if any document has None or empty content
        for doc in documents:
            if not doc.page_content or not doc.page_content.strip():
                st.warning(f"Skipping document with invalid content: {file_name}")
                continue

        # Create vector store for the document
        vectorstore = create_vectorstore_from_texts(documents, file_name)

        # Query the vector store using the user query
        structured_response = query_document(vectorstore, query=query)

        # Ensure structured response is a string and clean up any unwanted details
        if isinstance(structured_response, dict):
            response_text = str(structured_response.get('message', {}).get('content', ''))
        elif hasattr(structured_response, 'content'):
            response_text = str(structured_response.content)
        else:
            response_text = str(structured_response)

        # Append the extracted content (text) to the results
        results.append({
            "file_name": file_name,
            "answer": response_text
        })

        # Display the assistant's response as plain text
        with st.chat_message("assistant"):
            st.markdown(response_text)

        # Add assistant's response to session state (chat history) as clean text
        st.session_state["messages"].append({
            "role": "assistant",
            "content": response_text,
            "file_name": file_name  # You can still keep track of the file if needed
        })

def download_chat_history_to_csv():
    # Extract the chat history from session state (assuming it's stored like this)
    chat_history = st.session_state.get("messages", [])
    
    # Prepare lists to store inputs, outputs, and questions
    input_prompts = []
    output_responses = []
    questions = []

    # Iterate through the chat history to create the pairings
    user_input = None
    for entry in chat_history:
        if entry['role'] == 'user':
            # Store the user's input as the current user prompt
            user_input = entry.get('content', '')
        
        elif entry['role'] == 'assistant' and user_input:
            # Associate the previous user's input with the current assistant's output
            input_prompts.append(user_input)
            output_responses.append(entry.get('content', ''))
            questions.append(user_input)  # You can add more detailed logic for questions if needed
            
            # Reset the user_input for the next pairing
            user_input = None

    # Create a DataFrame
    df = pd.DataFrame({
        'questions': questions,
        'input_prompts': input_prompts,
        'output_responses': output_responses
    })
    
    # Convert the DataFrame to CSV format
    csv_data = df.to_csv(index=False)
    csv_buffer = io.StringIO(csv_data)
    
    # Trigger the download as a .csv file
    st.download_button(
        label="Download Chat History to CSV",
        data=csv_buffer.getvalue(),
        file_name="chat_history.csv",
        mime="text/csv"
    )