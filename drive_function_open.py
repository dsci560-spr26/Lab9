import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent
COLLECTION_NAME = "Ads_cookbook"
PERSIST_DIRECTORY = PROJECT_ROOT / "chroma_db_open"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")


def load_vectorstore():
    persist_directory = str(PERSIST_DIRECTORY)

    if not os.path.isdir(persist_directory):
        raise FileNotFoundError(
            f"Vector store directory not found: {persist_directory}. "
            f"Run preprocessing_open.py first."
        )

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME
    )

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=persist_directory,
        embedding_function=embeddings,
    )
    return vectorstore


def build_chain():
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

    llm = ChatOllama(
        model=OLLAMA_MODEL,
        temperature=0,
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
    )
    return chain


def main():
    print("Loading chatbot...")
    chain = build_chain()
    print("Chatbot ready. Type 'exit' to quit.\n")

    while True:
        question = input("You: ").strip()
        if not question:
            continue

        if question.lower() == "exit":
            print("Goodbye!")
            break

        try:
            result = chain.invoke({"question": question})
            answer = result.get("answer", "")
            print(f"\nBot: {answer}\n")

            source_documents = result.get("source_documents", [])
            if source_documents:
                print("Sources:")
                for doc in source_documents[:3]:
                    filename = doc.metadata.get("filename", "unknown")
                    page_num = doc.metadata.get("page_num", "unknown")
                    print(f"- {filename}, page {page_num}")
                print()

        except Exception as e:
            print(f"\nError: {e}\n")


if __name__ == "__main__":
    main()
