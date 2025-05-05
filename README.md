# Deepseek RAG-Assistant Pipeline

## Project Overview
The **Deepseek RAG-Assistant Pipeline** implements a robust conversational AI system that processes PDF documents and answers user queries in a dynamic, context-aware manner. This solution leverages **LangChain**, **Groq**, and **Chroma** to build a Retrieval-Augmented Generation (RAG) model that integrates document loading, text splitting, and conversational retrieval. It’s designed to handle and query large-scale document data efficiently.

### Key Features
- **Document Loading**: Seamlessly load and process PDF documents using `PyPDFLoader`.
- **Text Splitting**: Split large documents into manageable chunks using `RecursiveCharacterTextSplitter`.
- **Vectorization**: Convert text chunks into vector embeddings using the `HuggingFaceEmbeddings` model and store them in a `Chroma` vector store.
- **Conversational Retrieval Chain**: Build an interactive conversational assistant that provides relevant answers to questions based on the content of the document.
- **Groq Integration**: Utilize the `ChatGroq` model for generating responses, offering a high-performance, AI-driven conversational system.

## Requirements

To run this project, you need the following Python libraries:

```txt
opentelemetry-api==1.19.0
opentelemetry-sdk==1.19.0
opentelemetry-exporter-otlp-proto-grpc==1.19.0
langchain_community
pypdf
langchain_groq
chromadb
huggingface-hub
````

### Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/yourusername/deepseek-rag-assistant-pipeline.git
   cd deepseek-rag-assistant-pipeline
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

### Usage

1. **Set up Groq API Key**:

   * You'll need an API key for Groq. Set it in your environment variables:

   ```python
   os.environ["GROQ_API_KEY"] = "your-api-key-here"
   ```

2. **Load Documents**:

   * Provide the path to the PDF document you'd like to load.

   ```python
   file_path = "path_to_your_pdf_document.pdf"
   ```

3. **Run the Conversational Chain**:

   * After loading the document and splitting it into chunks, use the conversational chain to answer queries.

   ```python
   chain = main(file_path)
   response = chain.invoke({"question": "Give me the summary of the document?"})
   print(response["answer"])
   ```

### Example Workflow

1. Load a PDF document using `PyPDFLoader`.
2. Split the document into smaller text chunks.
3. Convert the text into vector embeddings using `HuggingFaceEmbeddings` and store them in a `Chroma` vector store.
4. Set up a conversational retrieval chain using `LangChain` and `Groq`.
5. Query the system with natural language questions, and the assistant will return context-based answers.

### Contributing

We welcome contributions! If you have suggestions, improvements, or bug fixes, feel free to submit issues or pull requests. Please follow the project's coding style and structure.
