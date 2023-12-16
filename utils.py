import re

import pinecone
from langchain.chains.summarize import load_summarize_chain
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.llms import HuggingFaceHub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
from pypdf import PdfReader
from pathlib import Path


def read_markdown_file(markdown_file):
    return Path(markdown_file).read_text()


# Extract Information from PDF file
def get_pdf_text(pdf_doc):
    text = ""
    pdf_reader = PdfReader(pdf_doc)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


# Read PDF data
def read_pdf_data(pdf_file):
    pdf_page = PdfReader(pdf_file)
    text = ""
    for page in pdf_page.pages:
        text += page.extract_text()
    return text


# Split data into chunks
def split_data(text, pdf_file):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    docs = text_splitter.split_text(text)

    metadata = {
        "name": pdf_file.name,
        "type=": pdf_file.type,
        "size": pdf_file.size,
    }
    docs_chunks = text_splitter.create_documents(docs, metadatas=[metadata] * len(docs))
    return docs_chunks


# Create embeddings instance
def create_embeddings_load_data():
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    return embeddings


# Function to push data to Pinecone
def push_to_pinecone(
    pinecone_apikey, pinecone_environment, pinecone_index_name, embeddings, docs
):
    pinecone.init(api_key=pinecone_apikey, environment=pinecone_environment)

    index_name = pinecone_index_name
    index = Pinecone.from_documents(docs, embeddings, index_name=index_name)
    return index


# Function to pull infrmation from Vector Store - Pinecone here
def pull_from_pinecone(
    pinecone_apikey, pinecone_environment, pinecone_index_name, embeddings
):
    pinecone.init(api_key=pinecone_apikey, environment=pinecone_environment)

    index_name = pinecone_index_name

    index = Pinecone.from_existing_index(index_name, embeddings)
    return index


# Function to help us get relevant documents from vector store - based on user input
def similar_docs(
    query, k, pinecone_apikey, pinecone_environment, pinecone_index_name, embeddings
):
    pinecone.init(api_key=pinecone_apikey, environment=pinecone_environment)

    index_name = pinecone_index_name

    index = pull_from_pinecone(
        pinecone_apikey, pinecone_environment, index_name, embeddings
    )
    similar_docs = index.similarity_search_with_score(query, int(k))
    return similar_docs


# Helps us get the summary of a document
def get_summary(current_doc):
    llm = HuggingFaceHub(
        repo_id="bigscience/bloom", model_kwargs={"temperature": 1e-10}
    )
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    summary = chain.run([current_doc])

    return summary


def parse_feedback_data(input_text):
    # Define a regular expression pattern to match the sections
    pattern = r"### (.*?) ###\nquestion: (.*?)\nanswer: (.*?)\nremark: (.*?)\n"

    matches = re.finditer(pattern, input_text)

    result = []

    for match in matches:
        topic = match.group(1)
        question = match.group(2)
        answer = match.group(3)
        remark = match.group(4)

        result.append(
            {
                "topic": topic,
                "question": question,
                "answer": answer,
                "remark": remark,
            }
        )

    # Print the parsed data as a dictionary
    return result