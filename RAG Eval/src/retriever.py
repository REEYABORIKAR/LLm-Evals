import os
import re
import glob
import time

from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
DB_DIR = os.path.join(BASE_DIR, "..", "chroma_store")

def load_transcripts():

    docs = []
    for path in glob.glob(os.path.join(DATA_DIR, "*.vtt")):
        lines = []
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if not line or line == "WEBVTT" or "-->" in line:
                continue
            lines.append(line)
        text = " ".join(lines)

        session = re.search(r"Session[ _]*(\d+)", os.path.basename(path)).group(1)

        docs.append(Document(page_content=text, metadata={"session": session}))

    if not docs:
        raise FileNotFoundError(f"No .vtt files found in {os.path.abspath(DATA_DIR)}")

    return docs


def load_store():
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

    if os.path.exists(DB_DIR) and os.listdir(DB_DIR):
        return Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

    docs = load_transcripts()

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    ).split_documents(docs)

    store = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

    batch_size = 90
    for i in range(0, len(chunks), batch_size):
        store.add_documents(chunks[i:i + batch_size])
        print(f"Embedded {min(i + batch_size, len(chunks))}/{len(chunks)}")
        if i + batch_size < len(chunks):
            time.sleep(65)

    return store


def build_retriever():
    return load_store().as_retriever(search_kwargs={"k": 5})


if __name__ == "__main__":

    retriever = build_retriever()

    results = retriever.invoke("what is regression testing?")
    
    for r in results:
        print(f"[Session {r.metadata['session']}] {r.page_content[:150]}...\n")