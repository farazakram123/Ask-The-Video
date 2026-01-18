import os
from dotenv import load_dotenv
from youtube_transcript_api import  TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
import streamlit as st
import requests
import json

load_dotenv()

def extract_video_id(url):
    # Extract video ID from YouTube URL
    if "v=" in url:
        return url.split("v=")[1].split("&")[0]
    elif "youtu.be/" in url:
        return url.split("youtu.be/")[1].split("?")[0]
    else:
        raise ValueError("Invalid YouTube URL")


def fetch_transcript(url):
    # Fetch transcript for a YouTube video
    endpoint = "https://api.supadata.ai/v1/youtube/transcript"
    
    headers = {
        "x-api-key": os.environ.get("SUPADATA_API_KEY") or st.secrets["SUPADATA_API_KEY"]
    }
    
    params = {
        "url": url,
        "text": "true"
    }

    try:
        response = requests.get(endpoint, headers=headers, params=params)
        response.raise_for_status() # Raises an error for 4xx or 5xx responses
        
        return response.json()['content']
    
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}
    
    except TranscriptsDisabled:
        raise Exception("No transcript available for this video.")
    except Exception as e:
        raise Exception(f"Error fetching transcript: {str(e)}")


def format_docs(retrieved_docs):
    """Format retrieved documents into a single string"""
    return "\n\n".join(doc.page_content for doc in retrieved_docs)


def process_video(url):
    # Process YouTube video and create RAG chain

    # 1. Extract video ID
    video_id = extract_video_id(url)

    # 2. Fetch Transcript
    transcript = fetch_transcript(url)

    # 3. Split Transcript into Chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript])

    # 4. Setup Embeddings
    embeddings = HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2"
    )
    index_name = os.environ.get("INDEX_NAME") or st.secrets["INDEX_NAME"]

    # 5. Initialize Pinecone Client
    pc = Pinecone(api_key= os.environ.get("PINECONE_API_KEY") or st.secrets["PINECONE_API_KEY"])

    # Delete everything in the "default" namespace
    if index_name in pc.list_indexes().names():
        index_ = pc.Index(index_name)
        index_.delete(delete_all=True, namespace="")

    # 6. Check if index exists; if not, create it
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=384,  # 384 for 'all-MiniLM-L6-v2'
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )

    # 7. Upload to Pinecone Vector Store
    vectorstore = PineconeVectorStore.from_documents(
        chunks,
        embeddings,
        index_name=index_name
    )

    # 8. Create a Retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 3})

    # 9. Create model
    llm = HuggingFaceEndpoint(
        repo_id="openai/gpt-oss-20b",
        task="text-generation",
        huggingfacehub_api_token = os.environ.get("HF_TOKEN") or st.secrets["HF_TOKEN"]
    )
    model = ChatHuggingFace(llm=llm)

    # 10. Create Prompt Template
    prompt = PromptTemplate(
        template="""You are a helpful assistant.
        Answer ONLY from the provided transcript context.
        If the context is insufficient, just say "This wasn't discussed here".

        {context}
        Question: {question}
        """,
                input_variables=["context", "question"]
    )

    # 11. Create Chains
    parallel_chain = RunnableParallel({
        'context': retriever | RunnableLambda(format_docs),
        'question': RunnablePassthrough()
    })

    parser = StrOutputParser()
    main_chain = parallel_chain | prompt | model | parser

    return main_chain


def ask_question(main_chain, question):
    # Ask a question using the RAG chain
    result = main_chain.invoke(question)
    return result