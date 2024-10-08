# RAG Chatbot with PDF Summarization

This project showcases the development of a **Retrieval Augmented Generation (RAG)** chatbot that interacts with a PDF document and summarizes large texts. It leverages **Langchain** and a large language model (LLM) like **GPT** or **Gemini** to provide accurate responses based on the document's content.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project 1: RAG Chatbot](#project-1-rag-chatbot)
- [Project 2: PDF Summarization](#project-2-pdf-summarization)
- [Screenshots](#screenshots)
- [License](#license)

## Project Overview

This repository contains two key projects:
1. **Building a RAG Chatbot**: A chatbot that answers questions based on the content of a PDF document. It uses **Langchain** and a vector database (ChromaDB) to retrieve relevant chunks from the document and respond effectively.
2. **PDF Summarization**: Summarizes the provided PDF document using various techniques such as `stuff`, `map_reduce`, and other chaining functions from Langchain.

## Features

- **Chatbot Functionality**: Answers queries based on the content of a PDF document.
- **Document Summarization**: Provides different types of summaries for large text documents, including concise and detailed summaries.
- **PDF Chunking**: Splits large documents into smaller, manageable chunks for efficient processing.
- **Embedding Vectors**: Generates embedding vectors for chunked documents using GPT or Gemini models.

## Installation

To set up and run the project locally, follow these steps:

1. **Clone the Repository**

    ```bash
    git clone https://github.com/yourusername/rag-chatbot-with-summarization.git
    cd rag-chatbot-with-summarization
    ```

2. **Create a Virtual Environment**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install Dependencies**

    ```bash
    pip install -r requirements.txt
    ```

4. **Run the Application**

    ```bash
    streamlit run app.py
    ```

5. Open your browser and navigate to `http://127.0.0.1:8501/` to interact with the chatbot.

## Usage

### Project 1: RAG Chatbot

1. **PDF Document Upload**: Upload a PDF document (e.g., [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://aclanthology.org/N19-1423.pdf)).
2. **Document Chunking**: The document is divided into smaller chunks for efficient processing.
3. **ChromaDB Setup**: ChromaDB is used to store the embedding vectors of the chunked documents.
4. **Embedding Vectors Creation**: Converts the chunked document into embedding vectors using GPT or Gemini models.
5. **Chatbot Interaction**: Users can ask questions, and the chatbot retrieves relevant information from ChromaDB to generate responses.

### Project 2: PDF Summarization

1. **PDF Upload and Chunking**: The PDF is uploaded and split into manageable chunks for summarization.
2. **Summarization Techniques**:
   - **Stuff Chain**: Generates a concise summary of the first 5 pages.
   - **Map Reduce Chain**: Produces a short summary of the entire document.
   - **Detailed Summary with Bullet Points**: Generates a detailed summary of the document using at least 1000 tokens.

## Screenshots

Here are some screenshots of the application:

*Figure 1: Uploading the PDF Document*  


*Figure 2: Chatbot Interaction Interface*  


*Figure 3: Summarization Output*  


## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

