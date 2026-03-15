import os
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains import ConversationalRetrievalChain

import chromadb

load_dotenv()

COLLECTION_NAME = "Ads_cookbook"
OPENAI_MODEL    = "gpt-4o-mini"
MAX_RETRIES     = 3


def get_chroma_cloud_client() -> chromadb.CloudClient:

    client = chromadb.CloudClient(
    api_key='ck-5dgKymXwyiqkzbwh67P8EcvqBoWEeVxa4GYjticy7ofC',
    tenant='b2a6f32d-669f-4c1a-8525-c857a7d1e59e',
    database='DSCI560_Lab9'
    )
    return client

def load_vector_store() -> Chroma:
    embeddings    = OpenAIEmbeddings(model="text-embedding-ada-002")
    chroma_client = get_chroma_cloud_client()

    vector_store = Chroma(
        client=chroma_client,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
    )
    print(f" Connected to Chroma Cloud collection '{COLLECTION_NAME}'.")
    return vector_store


def create_conversation_chain(llm: ChatOpenAI, vector_store: Chroma) -> ConversationalRetrievalChain:

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(search_kwargs={"k": 4}),
        memory=memory,
        return_source_documents=False,
        verbose=False,
    )
    print("Conversation chain ready.")
    return chain


# ===========================================================================
# e) DRIVER — interactive Q&A loop
# ===========================================================================

def confirm_question(question: str, llm: ChatOpenAI) -> str:
    """
    Ask the LLM to rephrase the user's question as a polite confirmation
    starting with "You're wondering".
    Does NOT answer the question — only echoes it back for confirmation.
    """
    prompt = (
        "Rephrase the following question as a polite one-sentence confirmation "
        "that starts with \"You're wondering\" and ends with a question mark. "
        "Do NOT answer the question — only echo it back as a confirmation.\n\n"
        f"Question: {question}"
    )
    return llm.invoke(prompt).content.strip()


def run_chatbot(chain: ConversationalRetrievalChain, llm: ChatOpenAI) -> None:
    """
    Interactive Q&A loop.

    Flow per question
    -----------------
    1. User types a question (or 'exit' to quit).
    2. LLM echoes: "You're wondering … — is that right? (yes/no)"
    3. 'yes'  → query Chroma Cloud → print answer → next question.
       other  → ask user to rephrase.
    4. After MAX_RETRIES (3) failed confirmations the session closes.
    """
    print("\nReady — type your question (or 'exit' to quit)\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        confirmed        = False
        current_question = user_input

        for attempt in range(1, MAX_RETRIES + 1):
            confirmation = confirm_question(current_question, llm)
            print(f"\nAssistant: {confirmation} — is that right? (yes/no)")
            user_confirm = input("You: ").strip().lower()

            if user_confirm == "yes":
                confirmed = True
                break

            # Not confirmed
            if attempt < MAX_RETRIES:
                remaining = MAX_RETRIES - attempt
                print(
                    f"\nAssistant: No problem — please rephrase your question. "
                    f"({remaining} attempt(s) remaining)\n"
                )
                current_question = input("You: ").strip()
                if current_question.lower() == "exit":
                    print("Goodbye!")
                    return
            else:
                print(
                    "\nAssistant: I'm sorry, I wasn't able to understand your question "
                    "after several attempts. The session will now close."
                )
                return

        if confirmed:
            result = chain.invoke({"question": current_question})
            answer = result.get("answer", "(No answer returned)")
            print(f"\nAssistant: {answer}\n")


def main() -> None:
    for var in ("OPENAI_API_KEY", "CHROMA_API_KEY", "CHROMA_TENANT", "CHROMA_DATABASE"):
        if not os.getenv(var):
            raise EnvironmentError(f"Environment variable '{var}' is not set.")

    print("\nLoading vector store & building conversation chain...")
    llm          = ChatOpenAI(model=OPENAI_MODEL, temperature=0)
    vector_store = load_vector_store()
    chain        = create_conversation_chain(llm, vector_store)

    print("\nStarting chatbot...")
    run_chatbot(chain, llm)


if __name__ == "__main__":
    main()


"""
Conversation Chain:   ConversationBufferMemory + ChatOpenAI + Chroma Cloud
Driver            :   Confirm-before-query loop (max 3 retries, type 'exit' to quit)

Environment variables required
-------------------------------
  OPENAI_API_KEY    your OpenAI API key
  CHROMA_API_KEY    your Chroma Cloud API key
  CHROMA_TENANT     your Chroma Cloud tenant name
  CHROMA_DATABASE   your Chroma Cloud database name

Usage
-----
  python chatbot.py
"""
