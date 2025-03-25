import streamlit as st
from llama_parse import LlamaParse, ResultType
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
import os
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
import nest_asyncio
import io
nest_asyncio.apply()



def store_in_faiss_db(langchain_documents):
    """
    Stores LangChain documents in a FAISS vector database.
    Uses session state instead of saving to disk.
    """
    embedding_model = OpenAIEmbeddings()

    # Use session state for FAISS to ensure isolation per user session
    if "faiss_db" in st.session_state:
        st.info(f"✅ Using existing FAISS DB in session")
        vector_db = st.session_state.faiss_db
    else:
        st.warning("⚠️ Creating new FAISS embeddings...")
        vector_db = FAISS.from_documents(langchain_documents, embedding_model)
        st.session_state.faiss_db = vector_db  # Store in session

    return vector_db





system_message = """"You are an expert in analyzing vendor documents. Your task is to answer user questions using only the content given to you. You will be provided with question and context.
   **Instructions for Answering User Queries:**

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

    **Output Format:**
        Provide a concise answer based on the context.
        When providing your final answer, strictly adhere to the answer Format:    <Answer>: ...  <Citation(s)>: ...  <Controls or Appendix>: ...  <Result>: Pass / Fail / Needs Additional Information.


        Answer: [Your concise response]
        Citation(s): [PDF name, page number, section, or specific text snippet]
        Controls or Appendix: [Relevant control numbers]
        Result: [Pass / Fail / Needs Additional Information]

    **Note:**
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
    """
    Converts parsed document objects into LangChain Document format.
    """
    langchain_docs = []

    for doc in parsed_files:
        langchain_doc = Document(page_content=doc.text, metadata=doc.metadata)
        langchain_docs.append(langchain_doc)

    return langchain_docs



def create_hybrid_retriever(vector_db, langchain_documents):
    if not langchain_documents:
        st.error("No documents to process!")
        return None

    
    for doc in langchain_documents:
        if not hasattr(doc, 'page_content') or not hasattr(doc, 'metadata'):
            st.error(f"Invalid document format: {doc}")
            return None

    vectorstore_retriever = vector_db.as_retriever(search_kwargs={"k": 8})

    try:
        keyword_retriever = BM25Retriever.from_documents(langchain_documents, k=4)
    except ValueError as e:
        st.error(f"Error in BM25Retriever: {e}")
        st.error(f"Documents: {langchain_documents}")
        return None

    ensemble_retriever = EnsembleRetriever(
        retrievers=[vectorstore_retriever, keyword_retriever],
        weights=[0.6, 0.4] 
    )

    return ensemble_retriever



st.set_page_config(page_title="RAG Vendor Documents", layout="wide")
st.title("📄 RAG Vendor Document Processor")
st.sidebar.header("⚙️ Configuration")

# Ensure API keys exist in session state
st.session_state.setdefault("llama_api_key", "")
st.session_state.setdefault("openai_api_key", "")

# Input for API keys
llama_api_key = st.sidebar.text_input("Llama Cloud API Key", type="password")
openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")

# Update session state only if a new key is provided
if llama_api_key:
    st.session_state.llama_api_key = llama_api_key
    os.environ["LLAMA_CLOUD_API_KEY"] = llama_api_key  # Set environment variable

if openai_api_key:
    st.session_state.openai_api_key = openai_api_key
    os.environ["OPENAI_API_KEY"] = openai_api_key  # Set environment variable

# Stop execution if keys are missing
if not st.session_state.llama_api_key or not st.session_state.openai_api_key:
    st.error("❌ Please enter both Llama Cloud API Key and OpenAI API Key before uploading documents.")
    st.stop()




if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = {}

uploaded_files = st.sidebar.file_uploader("Upload vendor documents", accept_multiple_files=True, type="pdf")

if uploaded_files:
    for uploaded_file in uploaded_files:
        file_bytes = uploaded_file.getvalue()  # Get binary content
        st.session_state.uploaded_files[uploaded_file.name] = file_bytes  # Store in session




def load_and_process_documents(uploaded_files):
    """
    Parses documents from uploaded files and stores them in session state.
    """
    system_prompt_append = """
#     If a row in table does not contain a value in the No. column, use the last available No. from the previous rows in the same table.
#     If a table starts without an explicit No. in the first row, use the last No. from the previous table.
#     If table's first row does not contain a value in the No. column, then use the last value seen to populate it.
#     """

    parser = LlamaParse(result_type=ResultType.MD, system_prompt_append=system_prompt_append, parsing_mode="parse_document_with_llm")

    if "parsed_files" not in st.session_state:
        st.session_state.parsed_files = {}

    parsed_files = []

    for file_name, file_bytes in uploaded_files.items():
        if file_name in st.session_state.parsed_files:
            st.write(f"✅ **Skipping already parsed file:** {file_name}")
            parsed_files.extend(st.session_state.parsed_files[file_name])
            continue  # Skip re-parsing

        try:
            st.write(f"📄 **Processing file:** {file_name}")
            file_stream = io.BytesIO(file_bytes)

            # Parse the document
            documents = parser.load_data(file_stream, extra_info={"file_name": file_name})
            for doc in documents:
                doc.metadata["file_name"] = file_name

            # Store parsed documents
            st.session_state.parsed_files[file_name] = documents
            parsed_files.extend(documents)

            st.write(f"✅ Successfully parsed {len(documents)} documents from {file_name}.")

        except Exception as e:
            st.error(f"❌ Error processing {file_name}: {e}")

    return parsed_files




# Process the documents after uploading
if st.session_state.uploaded_files:
    parsed_docs = load_and_process_documents(st.session_state.uploaded_files)

    # Store parsed documents in session
    st.session_state.parsed_docs = parsed_docs


if uploaded_files:
    # Store uploaded files in session state instead of saving them to disk
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = {}

    for uploaded_file in uploaded_files:
        file_bytes = uploaded_file.getvalue()  # Get binary content
        if uploaded_file.name not in st.session_state.uploaded_files:
            st.session_state.uploaded_files[uploaded_file.name] = file_bytes  # Store in session

    # 🛠 Process documents only if they haven't been processed before
    if "parsed_docs" not in st.session_state:
        parsed_files = load_and_process_documents(st.session_state.uploaded_files)
        st.session_state.parsed_docs = parsed_files  # Store parsed docs
    else:
        parsed_files = st.session_state.parsed_docs

    # Convert parsed documents into LangChain format
    langchain_documents = convert_to_langchain_documents(parsed_files)

    if langchain_documents:
        st.session_state.langchain_documents = langchain_documents  # Store in session

        st.success(f"✅ Converted {len(langchain_documents)} documents into LangChain format!")

        st.write(f"📝 **Total Parsed Files:** {len(parsed_files)}")
        st.write(f"📄 **Total LangChain Documents:** {len(langchain_documents)}")

    # Validate documents
    all_docs_valid = all(isinstance(doc, Document) for doc in langchain_documents)
    st.write(f"✅ **All files converted correctly:** {all_docs_valid}")

    # Store documents in FAISS (User-specific)
    vector_db = store_in_faiss_db(langchain_documents)
    st.session_state.vector_db = vector_db  # Store in session

    # Create a hybrid retriever using stored FAISS
    retriever = create_hybrid_retriever(vector_db, langchain_documents)
    st.session_state.retriever = retriever  # Store in session


if "retriever" in st.session_state:
    st.write("## 🔎 Ask Multiple Questions")
    user_queries = st.text_area("Enter multiple queries (one per line):")

    if user_queries:
        queries = user_queries.strip().split("\n")  # Split input into separate queries

        for i, query in enumerate(queries, start=1):
            st.write(f"### 🔹 Query {i}: {query}")  
            response = answer_query(query, st.session_state.retriever, system_message)  # Use session retriever
            clean_response = response.content if hasattr(response, "content") else str(response)  # Extract response text

            st.success(clean_response)
