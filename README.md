# FDA Cosmetic Advisor Chatbot
![Cosmetic Advisor Chatbot](../Cosmetic_advisor_chatbot.png)
This project is an interactive chatbot that provides answers to user queries based on FDA guidelines for cosmetics. The chatbot processes uploaded PDF files, extracts relevant information, and uses an AI model to answer questions based on the content of those documents. The assistant provides answers, tracks the chat history, and allows users to download the chat history as a CSV file.

## Features

- **Upload FDA guidelines**: Users can upload multiple PDF files containing FDA guidelines on cosmetics.
- **Question answering**: The assistant can answer questions about the uploaded guidelines based on the content in the PDFs.
- **Chat history**: The app stores the entire chat history, showing both user and assistant messages.
- **Download chat history**: Users can download the full chat history (questions and responses) in a CSV format.

## Requirements

To run this project, you need the following Python libraries:

- `langchain`
- `langchain_ollama`
- `streamlit`
- `pandas`
- `pydantic`
- `Chroma`
- `uuid`
- `re`
- `os`
- `io`
- `tempfile`
- `base64`

Install these dependencies by running:

```bash
pip install langchain langchain-ollama streamlit pandas pydantic chromadb
```

## Files Overview

### 1. **functions.py**

This script defines the main functions for processing the PDF files, extracting text, generating embeddings, creating a vector store, and answering user queries. The functions are used by the main app script to process user inputs and PDF uploads.

#### Key Functions:
- **`get_pdf_text(uploaded_file)`**: Reads the content of uploaded PDF files.
- **`split_document(documents, chunk_size, chunk_overlap)`**: Splits documents into text chunks for easier processing.
- **`create_vectorstore(chunks, embedding_function, file_name, vector_store_path="db")`**: Creates a Chroma vector store to hold the text chunks and embeddings.
- **`process_multiple_pdfs(pdf_files, query)`**: Processes multiple PDFs, extracts text, creates vector stores, and generates answers based on the user's query.
- **`query_document(vectorstore, query)`**: Queries the vector store with the provided query and returns a structured response.

### 2. **app.py**

This is the Streamlit web interface for interacting with the chatbot. It provides a user-friendly environment where users can upload PDFs, input queries, and see the assistant’s responses.

#### Key Features:
- **File Upload**: Users can upload FDA guidelines PDFs for processing.
- **Chat Interface**: Users can input questions, see the assistant’s responses, and interact with the chatbot in real-time.
- **Chat History**: The entire chat history is stored and displayed, allowing users to view past interactions.
- **Clear History**: Users can clear the chat history.
- **Download Chat History**: Users can download the entire chat history (questions and responses) as a CSV file.

### Workflow:
1. **Upload PDFs**: The user uploads one or more PDF files containing FDA guidelines.
2. **Ask a Question**: The user types a question about the uploaded documents.
3. **Process the Query**: The app processes the PDFs and retrieves relevant context, using the LangChain pipeline to generate an answer.
4. **View Responses**: The assistant provides an answer based on the context of the uploaded PDFs.
5. **Chat History**: All interactions are saved in the chat history, which can be downloaded as a CSV file.
6. **Clear History**: The user can reset the chat history at any time.

### Example Workflow:
1. **Upload PDF files**: The user uploads FDA guideline PDFs.
2. **Enter Query**: The user types something like "What are the guidelines for cosmetics packaging?"
3. **Assistant Response**: The chatbot processes the PDFs, finds relevant information, and provides an answer.
4. **Download History**: After the session, the user can download the entire chat history as a CSV file.

## Usage Instructions

1. **Running the App**:
   To start the Streamlit app, run the following command:

   ```bash
   streamlit run app.py
   ```

   This will open the app in your browser.

2. **Interacting with the Chatbot**:
   - **Upload PDF documents**: Click on the "Use this section to upload FDA guidelines" expander to upload your PDF files.
   - **Ask a question**: Type your question in the chat input box and click enter.
   - **View the assistant's response**: The chatbot will provide an answer based on the PDF contents.
   - **Clear chat history**: Click the "Clear Chat History" button to reset the session.
   - **Download chat history**: After interacting with the chatbot, click "Download Chat History to CSV" to save your questions and responses.

3. **Download Chat History as CSV**:
   Once you are done with the session, you can download the chat history (questions and responses) by clicking the "Download Chat History to CSV" button. The data will be downloaded in CSV format, which you can open in any spreadsheet application.

## Customization

- Modify the **`PROMPT_TEMPLATE`** in `functions.py` to change how the assistant responds to queries.
- Customize **chunk size** and **overlap** in `split_document` to adjust document processing behavior.
- Change the **embedding model** in `get_embedding_function` to experiment with different AI models.

## Troubleshooting

- If you encounter issues with uploading or processing PDF files, make sure the files are not empty and are in the correct format (`.pdf`).
- If the chat history seems mismatched, make sure both user inputs and assistant responses are correctly paired in the chat.

## Conclusion

This project provides an interactive tool for querying FDA guidelines for cosmetics, using an AI-powered chatbot to deliver insights directly from uploaded documents. It’s a useful resource for anyone needing quick, context-based answers to questions about cosmetic regulations.