import streamlit as st
from pypdf import PdfReader  # New package name
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import traceback

# Load environment variables
load_dotenv()
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

if not google_api_key:
    st.error("GOOGLE_API_KEY not found in environment variables")
    st.stop()
genai.configure(api_key=google_api_key)

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:  # Only add if text was extracted
                    text += page_text
        except Exception as e:
            st.error(f"Error reading PDF {pdf.name}: {str(e)}")
            continue
    return text

def get_text_chunks(text):
    if not text.strip():
        return []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000, 
        chunk_overlap=1000
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    if not text_chunks:
        raise ValueError("No text chunks provided for vector store creation")
    
    # Updated to use the newer text-embedding-004 model
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    try:
        # Updated to use the more capable gemini-1.5-pro-latest model
        model = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro-latest",
            temperature=0.3
        )
    except Exception as e:
        st.error(f"Failed to initialize Gemini model: {str(e)}")
        st.error("Please check your API key and ensure the model is available")
        raise

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    try:
        if not os.path.exists("faiss_index"):
            st.error("Vector store not found. Please process PDFs first.")
            return

        # Updated to use the newer text-embedding-004 model
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        
        # Load FAISS with dangerous deserialization allowed (trusted source only)
        new_db = FAISS.load_local(
            "faiss_index", 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        
        docs = new_db.similarity_search(user_question)
        chain = get_conversational_chain()
        
        # Using invoke() instead of __call__() to avoid deprecation warning
        response = chain.invoke(
            {"input_documents": docs, "question": user_question}
        )
        
        st.write("Reply:", response["output_text"])
        
    except Exception as e:
        st.error(f"An error occurred while processing your question: {str(e)}")
        st.error("Please try again or upload your documents once more.")

def main():
    st.set_page_config("Chat PDF", page_icon="📄")
    st.header("Chat with PDF using Gemini 💁")
    
    # Warning about dangerous deserialization
    st.warning(
        "Note: This application processes PDFs you upload. "
        "Only upload documents from trusted sources."
    )

    user_question = st.text_input("Ask a question about your PDF documents:")
    
    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu")
        st.markdown("""
        ### Instructions:
        1. Upload your PDF files
        2. Click 'Submit & Process'
        3. Ask questions about your documents
        """)
        
        pdf_docs = st.file_uploader(
            "Upload PDF Files",
            accept_multiple_files=True,
            type=["pdf"]
        )
        
        if st.button("Submit & Process"):
            if not pdf_docs:
                st.warning("Please upload at least one PDF file.")
                return

            with st.spinner("Processing your documents..."):
                try:
                    # Extract text
                    raw_text = get_pdf_text(pdf_docs)
                    
                    if not raw_text.strip():
                        st.error("No text could be extracted from the PDF(s). They might be image-based or empty.")
                        return
                    
                    # Split into chunks
                    text_chunks = get_text_chunks(raw_text)
                    
                    if not text_chunks:
                        st.error("Failed to split the text into meaningful chunks.")
                        return
                    
                    # Create and store vector store
                    get_vector_store(text_chunks)
                    st.success("Processing complete! You can now ask questions about your documents.")
                    
                except Exception as e:
                    st.error(f"Processing failed: {str(e)}")
                    st.error("Please check your documents and try again.")

if __name__ == "__main__":
    main()
