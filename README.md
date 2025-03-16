# FDA Cosmetic Advisor Chatbot  
![Cosmetic_advisor_chatbot_eyeliner](https://github.com/user-attachments/assets/8129507a-3a8a-4e7b-9236-9e05d7c097a0)  

This project is an interactive chatbot that provides answers to user queries based on FDA guidelines for cosmetics. The chatbot processes uploaded PDF files, extracts relevant information, and uses an AI model to answer questions based on the content of those documents. The assistant provides answers, tracks the chat history, and allows users to download the chat history as a CSV file.  

## Features  

- **Upload FDA guidelines**: Users can upload multiple PDF files containing FDA guidelines on cosmetics.  
- **Question answering**: The assistant can answer questions about the uploaded guidelines based on the content in the PDFs.  
- **Chat history**: The app stores the entire chat history, showing both user and assistant messages.  
- **Download chat history**: Users can download the full chat history (questions and responses) in a CSV format.  

## Requirements  

To run this project, you need the following dependencies installed:  

### Install Python Packages  

```bash
pip install -r requirements.txt
```

### Download the `nomic-embed-text` Model for Ollama  

The chatbot requires `nomic-embed-text` for embedding functionality. To download it, run:  

```bash
ollama pull nomic-embed-text
```

You can verify that `ollama` is correctly set up by running:  

```bash
ollama list
```

## Files Overview  

### 1. **functions.py**  

This script defines the main functions for processing the PDF files, extracting text, generating embeddings, creating a vector store, and answering user queries.  

#### Key Functions:  
- **`get_pdf_text(uploaded_file)`**: Reads the content of uploaded PDF files.  
- **`split_document(documents, chunk_size, chunk_overlap)`**: Splits documents into text chunks for easier processing.  
- **`create_vectorstore(chunks, embedding_function, file_name, vector_store_path="db")`**: Creates a Chroma vector store to hold the text chunks and embeddings.  
- **`process_multiple_pdfs(pdf_files, query)`**: Processes multiple PDFs, extracts text, creates vector stores, and generates answers based on the user's query.  
- **`query_document(vectorstore, query)`**: Queries the vector store with the provided query and returns a structured response.  

### 2. **streamlit_app.py**  

This is the Streamlit web interface for interacting with the chatbot. It provides a user-friendly environment where users can upload PDFs, input queries, and see the assistant’s responses.  

#### Key Features:  
- **File Upload**: Users can upload FDA guidelines PDFs for processing.  
- **Chat Interface**: Users can input questions, see the assistant’s responses, and interact with the chatbot in real time.  
- **Chat History**: The entire chat history is stored and displayed, allowing users to view past interactions.  
- **Clear History**: Users can clear the chat history.  
- **Download Chat History**: Users can download the entire chat history (questions and responses) as a CSV file.  

## Usage Instructions  

### Running the App  

To start the Streamlit app, run:  

```bash
streamlit run streamlit_app.py
```

This will open the app in your browser.  

### Interacting with the Chatbot  

1. **Upload PDF documents**: Click on the upload section and select FDA guideline PDFs.  
2. **Ask a question**: Type a query and hit enter.  
3. **View responses**: The chatbot will answer based on the uploaded PDFs.  
4. **Download chat history**: Click the "Download Chat History to CSV" button to save your session.  
5. **Clear chat history**: Reset the session anytime by clicking "Clear Chat History."  

### Sample PDF  

To quickly test the chatbot, you can use this sample FDA guideline PDF:  

[Download Sample FDA Cosmetic Guidelines PDF](https://www.fda.gov/media/88234/download)  

## Troubleshooting  

- If the chatbot does not return answers, ensure you have run `ollama pull nomic-embed-text`.  
- If PDF uploads fail, ensure the files are valid PDFs.  
- If chat history does not download, check your browser’s download permissions.  

## Conclusion  

This project provides an interactive tool for querying FDA guidelines for cosmetics using an AI-powered chatbot. It helps users get quick answers to regulatory questions efficiently.  

## Contributing  
Pull requests are welcome. For major changes, please open an issue first to discuss what you'd like to change.  

## License  

This project is licensed under the [MIT License](LICENSE).  
