import streamlit as st
from llama_parse import LlamaParse, ResultType
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
from typing import List
import os
import pickle
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_openai import ChatOpenAI
import asyncio
import nest_asyncio
nest_asyncio.apply()


system_message = """You are an expert in analyzing vendor documents. Your task is to answer user questions using only the content given to you. You will be provided with question and context.
   ### **Instructions for Answering User Queries:**

    1. **Review the Question:**  
      - Analyze the user's question and determine the required information.

    2. **Check for Answer in the Provided Context:**  
      - Search for all relevant chunks in the provided context that directly answer the question.
      - Extract the exact PDF name and page number where the relevant information is found.

    3. **Citation & Control Numbers Extraction:**
      - If the context contains relevant information:
        - Extract **all matching PDF names and their page numbers** where the answer is found.
        - Extract **all control numbers** that fully satisfy the question.
      
    4. **If No Sufficient Information is Found:**
      - Respond with **“Needs Additional Information”** instead of guessing.

    ### **Output Format:**
        Provide a concise answer based on the context.
        When providing your final answer, strictly adhere to the answer Format:    <Answer>: ...  <Citation(s)>: ...  <Controls or Appendix>: ...  <Result>: Pass / Fail / Needs Additional Information.


        Answer: [Your concise response]
        Citation(s): [PDF name, page number, section, or specific text snippet]
        Controls or Appendix: [Relevant control numbers]
        Result: [Pass / Fail / Needs Additional Information]

    ### **Note:**
      citation number can be anything like (CC 1.2, CC5.3, PI 1.2, A2.1, etc) combination of alpha-numeric.
      Before adding citation, make sure that the information is correct.
      There can be more than one relevant pdf, add all relevant pdf name and respective page number in citation section seperated by ",".
      DO NOT halucinate with wrong infromation.
      Only use context to answer question, do not use your own knowledge
      If multiple documents contain relevant or overlapping information, **prioritize the document with “SOC_2” in the file name** while still listing others.

    """

def answer_query(query, retriever,system_prompt):
    # Retrieve relevant document chunks
    hybrid_chunks = retriever.invoke(query)
    context = ""

    for chunk in hybrid_chunks:
        file_name = chunk.metadata.get("file_name", "Unknown")  # Extract file name
        context += f"📄 **File:** {file_name}\n🔹 **Content:** {chunk.page_content}\n\n."

    # Format prompt for the LLM
    system_message = system_prompt
    human_message = f"🔎 **Query:** {query}\n\n📜 **Context:** {context} "

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": human_message}
    ]

    # Call LLM to generate answer
    llm = ChatOpenAI(model="gpt-4o",temperature=0)
    response = llm.invoke(messages)

    return response

def convert_to_langchain_documents(parsed_files):
    langchain_docs = []
    for file_path in parsed_files:
        with open(file_path, "rb") as file:  
            documents = pickle.load(file)  

        
        for doc in documents:
            langchain_doc = Document(page_content=doc.text, metadata=doc.metadata)
            langchain_docs.append(langchain_doc)

    return langchain_docs

def create_hybrid_retriever(vector_db, langchain_documents):
    vectorstore_retriever = vector_db.as_retriever(search_kwargs={"k": 8})
    keyword_retriever = BM25Retriever.from_documents(langchain_documents, k=4)

    ensemble_retriever = EnsembleRetriever(
        retrievers=[vectorstore_retriever, keyword_retriever],
        weights=[0.6, 0.4]  
    )

    return ensemble_retriever

def store_in_chroma_db(langchain_documents):
    chroma_db_path = "chroma_db"  
    embedding_model = OpenAIEmbeddings()

    if os.path.exists(chroma_db_path):
        vector_db = Chroma(persist_directory=chroma_db_path, embedding_function=embedding_model)
        st.info(f"✅ Loaded existing ChromaDB from `{chroma_db_path}`")
    else:
        st.warning("⚠️ ChromaDB not found! Creating new embeddings...")
        vector_db = Chroma.from_documents(langchain_documents, embedding_model, persist_directory=chroma_db_path)
        st.success(f"✅ ChromaDB created and saved at `{chroma_db_path}`")

    # Verify stored document count
    num_vectors = vector_db._collection.count()
    st.write(f"🔢 **Total documents indexed in ChromaDB:** {num_vectors}")

    return vector_db

st.set_page_config(page_title="RAG Vendor Documents", layout="wide")
st.title("📄 RAG Vendor Document Processor")
st.sidebar.header("⚙️ Configuration")

llama_api_key = st.sidebar.text_input("Llama Cloud API Key", type="password")
openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")

if llama_api_key:
    os.environ["LLAMA_CLOUD_API_KEY"] = llama_api_key
if openai_api_key:
    os.environ["OPENAI_API_KEY"] = openai_api_key


uploaded_files = st.sidebar.file_uploader("Upload vendor documents", accept_multiple_files=True, type="pdf")




def load_and_process_documents(file_paths: List[str], output_dir: str):
    """
    Parses documents using LlamaParse and saves them as Pickle files.
    Uses caching to avoid re-parsing if already processed.
    """
    st.write("📌 **Parsing documents...**")

    system_prompt_append = """
    If a row in table does not contain a value in the No. column, use the last available No. from the previous rows in the same table.
    If a table starts without an explicit No. in the first row, use the last No. from the previous table.
    If table's first row does not contain a value in the No. column, then use the last value seen to populate it.
    """

    parser = LlamaParse(result_type=ResultType.MD, system_prompt_append=system_prompt_append, parsing_mode="parse_document_with_llm")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    parsed_files = []
    progress_bar = st.progress(0)  # Streamlit progress bar

    for idx, file_path in enumerate(file_paths):
        file_name = os.path.basename(file_path)
        output_file = os.path.join(output_dir, f"parsed_{file_name}.pkl")

        if os.path.exists(output_file):
            st.info(f"✅ Loading cached document: **{file_name}**")
            with open(output_file, 'rb') as file:
                documents = pickle.load(file)
        else:
            st.warning(f"⚠️ Parsing document: **{file_name}** ...")
            try:
                documents = parser.load_data(file_path)  # ✅ No async handling needed

                # Add metadata
                for doc in documents:
                    if not hasattr(doc, "metadata"):
                        doc.metadata = {}
                    doc.metadata["file_name"] = file_name

                # Save parsed document
                with open(output_file, 'wb') as file:
                    pickle.dump(documents, file)

            except Exception as e:
                st.error(f"❌ Error while parsing '{file_name}': {e}")
                continue

        parsed_files.append(output_file)
        progress_bar.progress((idx + 1) / len(file_paths))
    st.success("✅ All documents processed successfully!")
    return parsed_files

if uploaded_files:
    output_dir = "processed_docs"
    os.makedirs(output_dir, exist_ok=True)

    # Save uploaded files to local directory
    file_paths = []
    for uploaded_file in uploaded_files:
        file_path = os.path.join(output_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        file_paths.append(file_path)

    # 🛠 Call the function to process documents
    parsed_files = load_and_process_documents(file_paths, output_dir)

    langchain_documents = convert_to_langchain_documents(parsed_files)

    st.success(f"✅ Converted {len(langchain_documents)} documents into LangChain format!")

    st.write(f"📝 **Total Parsed Files:** {len(parsed_files)}")
    st.write(f"📄 **Total LangChain Documents:** {len(langchain_documents)}")

    # Validate documents
    all_docs_valid = all(isinstance(doc, Document) for doc in langchain_documents)
    st.write(f"✅ **All files converted correctly:** {all_docs_valid}")


    # Store documents in ChromaDB
    vector_db = store_in_chroma_db(langchain_documents)


    retriever = create_hybrid_retriever(vector_db, langchain_documents)


    st.write("## 🔎 Ask Multiple Questions")
    user_queries = st.text_area("Enter multiple queries (one per line):")

    if user_queries:
        queries = user_queries.strip().split("\n")  # Split input into separate queries

        for i, query in enumerate(queries, start=1):
            st.write(f"### 🔹 Query {i}: {query}")  
            response = answer_query(query, retriever, system_message)  # Process each query
            st.success(response)  # Show the response
