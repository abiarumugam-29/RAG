import os
import uuid
import streamlit as st

from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI


# -------------------- PAGE SETUP --------------------

st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="📄",
    layout="wide",
)

st.title("📄 PDF RAG Chatbot")
st.caption("Ask questions using only the content of your uploaded PDF.")


# -------------------- SESSION STATE --------------------

defaults = {
    "chats": {},
    "active_chat": None,
    "vectorstore": None,
    "pdf_stats": {},
    "pdf_id": None,
    "api_key": "",
    "processed_file": None,
    "embedding_model": None,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# -------------------- HELPER FUNCTIONS --------------------

FALLBACK = "Sorry, I couldn't find the answer in the uploaded PDF."


def create_chat():
    chat_id = str(uuid.uuid4())
    st.session_state.chats[chat_id] = {
        "title": "New Chat",
        "messages": [],
        "pdf_id": st.session_state.pdf_id,
    }
    st.session_state.active_chat = chat_id


def reset_document_context():
    st.session_state.vectorstore = None
    st.session_state.pdf_stats = {}
    st.session_state.pdf_id = None
    st.session_state.processed_file = None
    st.session_state.chats = {}
    st.session_state.active_chat = None


def ask_pdf(question):
    store = st.session_state.vectorstore
    if store is None:
        return FALLBACK, []

    # Chroma distance: lower is generally more similar.
    matches = store.similarity_search_with_score(question, k=4)

    # Initial threshold based on the notebook's test.
    relevant_docs = [
        doc for doc, distance in matches
        if distance < 1.0
    ]

    if not relevant_docs:
        return FALLBACK, []

    context = "\n\n".join(
        "[Page " + str(doc.metadata.get("page", "Unknown")) + "]\n"
        + doc.page_content
        for doc in relevant_docs
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        google_api_key=st.session_state.api_key,
    )

    prompt = f"""
You are a PDF question-answering assistant.

Rules:
1. Answer using ONLY the supplied PDF context.
2. Do not use outside knowledge.
3. If the answer is not in the context, reply exactly:
{FALLBACK}
4. Be clear and concise.
5. Do not invent facts or page numbers.

PDF context:
{context}

Question:
{question}

Answer:
"""

    response = llm.invoke(prompt)
    answer = response.content

    if isinstance(answer, list):
        text_parts = []
        for item in answer:
            if isinstance(item, dict) and item.get("text"):
                text_parts.append(item["text"])
        answer = "\n".join(text_parts)

    answer = str(answer).strip()

    pages = sorted(
        set(
            doc.metadata.get("page")
            for doc in relevant_docs
            if doc.metadata.get("page") is not None
        )
    )

    if answer == FALLBACK:
        return FALLBACK, []

    if pages:
        page_text = ", ".join(str(page) for page in pages)
        answer += "\n\n**Source pages:** " + page_text

    return answer, pages


# -------------------- SIDEBAR --------------------

with st.sidebar:
    st.header("📚 Chat History")

    if st.button("➕ New Chat", use_container_width=True):
        create_chat()
        st.rerun()

    if not st.session_state.chats:
        st.caption("Your chats will appear here.")

    for chat_id, chat in list(st.session_state.chats.items()):
        col1, col2 = st.columns([5, 1])

        title = chat.get("title", "New Chat")

        if col1.button(
            title,
            key="open_" + chat_id,
            use_container_width=True,
        ):
            st.session_state.active_chat = chat_id
            st.rerun()

        if col2.button("🗑️", key="delete_" + chat_id):
            del st.session_state.chats[chat_id]

            if st.session_state.active_chat == chat_id:
                st.session_state.active_chat = None

            st.rerun()

    st.divider()
    st.subheader("📄 Upload PDF")

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        key="pdf_uploader",
    )

    api_key_input = st.text_input(
        "Gemini API key",
        type="password",
        value=st.session_state.api_key,
        help="For deployment, configure GOOGLE_API_KEY in Render environment variables.",
    )

    if api_key_input.strip():
        st.session_state.api_key = api_key_input.strip()
    elif os.environ.get("GOOGLE_API_KEY"):
        st.session_state.api_key = os.environ["GOOGLE_API_KEY"]

    process_clicked = st.button(
        "⚙️ Process PDF",
        use_container_width=True,
    )

    if uploaded_file is not None:
        current_file_id = (
            uploaded_file.name,
            uploaded_file.size,
        )

        if (
            st.session_state.processed_file is not None
            and current_file_id != st.session_state.processed_file
        ):
            reset_document_context()
            st.info("New PDF detected. Process it to start a fresh chat.")


# -------------------- PROCESS PDF --------------------

if process_clicked:
    if uploaded_file is None:
        st.error("Please upload a PDF first.")
    elif not st.session_state.api_key:
        st.error("Please provide your Gemini API key.")
    else:
        try:
            with st.spinner("Reading PDF and creating embeddings..."):
                reader = PdfReader(uploaded_file)

                all_pages = []
                documents = []

                for page_num, page in enumerate(reader.pages, start=1):
                    text = page.extract_text() or ""
                    all_pages.append(text)

                    if text.strip():
                        documents.append(
                            Document(
                                page_content=text,
                                metadata={
                                    "page": page_num,
                                    "source": uploaded_file.name,
                                },
                            )
                        )

                if not documents:
                    st.error(
                        "No selectable text found. This may be a scanned PDF."
                    )
                    st.stop()

                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=800,
                    chunk_overlap=150,
                )

                chunks = splitter.split_documents(documents)

                if not chunks:
                    st.error("No text chunks were created.")
                    st.stop()

                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )

                store = Chroma.from_documents(
                    documents=chunks,
                    embedding=embeddings,
                    collection_name="pdf_" + uuid.uuid4().hex,
                )

                st.session_state.vectorstore = store
                st.session_state.embedding_model = embeddings
                st.session_state.pdf_id = (
                    uploaded_file.name,
                    uploaded_file.size,
                )
                st.session_state.processed_file = (
                    uploaded_file.name,
                    uploaded_file.size,
                )

                st.session_state.pdf_stats = {
                    "filename": uploaded_file.name,
                    "pages": len(reader.pages),
                    "pages_with_text": len(documents),
                    "chunks": len(chunks),
                    "chunk_size": 800,
                    "chunk_overlap": 150,
                    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                }

                # A newly processed PDF starts fresh.
                st.session_state.chats = {}
                create_chat()

            st.success("PDF processed successfully!")
            st.rerun()

        except Exception as error:
            st.error("PDF processing failed.")
            st.exception(error)


# -------------------- PDF ANALYTICS --------------------

if st.session_state.pdf_stats:
    stats = st.session_state.pdf_stats

    st.subheader("📊 PDF Analytics")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total pages", stats["pages"])
    col2.metric("Pages with text", stats["pages_with_text"])
    col3.metric("Total chunks", stats["chunks"])

    with st.expander("View processing details"):
        st.write("**Filename:**", stats["filename"])
        st.write("**Chunk size:**", stats["chunk_size"])
        st.write("**Chunk overlap:**", stats["chunk_overlap"])
        st.write("**Embedding model:**", stats["embedding_model"])
        st.write("**Retrieved chunks per query:** up to 4")
        st.write("**Initial distance threshold:** 1.0")


# -------------------- CHAT INTERFACE --------------------

if st.session_state.vectorstore is None:
    st.info("Upload a PDF, enter your Gemini API key, and click Process PDF.")

elif st.session_state.active_chat is None:
    st.info("Click New Chat to start asking questions.")

elif st.session_state.active_chat in st.session_state.chats:
    chat = st.session_state.chats[st.session_state.active_chat]

    st.subheader(chat.get("title", "PDF Chat"))

    # Display previous messages.
    for message in chat["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Ask something about your PDF...")

    if question:
        chat["messages"].append({
            "role": "user",
            "content": question,
        })

        if chat["title"] == "New Chat":
            chat["title"] = question[:35]

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching the PDF..."):
                try:
                    answer, pages = ask_pdf(question)
                    st.markdown(answer)

                except Exception as error:
                    answer = "An error occurred while generating the answer."
                    st.error(answer)
                    st.exception(error)

        chat["messages"].append({
            "role": "assistant",
            "content": answer,
        })

