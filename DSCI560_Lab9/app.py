from pathlib import Path
import sys

import streamlit as st
from dotenv import load_dotenv

from htmlTemplates import css, bot_template, user_template


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from chat_backend import MODE_OPTIONS, build_chain, process_pdfs


def render_chat_history():
    for message in st.session_state.display_messages:
        template = user_template if message["role"] == "user" else bot_template
        st.write(
            template.replace("{{MSG}}", message["content"]),
            unsafe_allow_html=True,
        )


def handle_userinput(user_question: str, chat_container):
    if st.session_state.conversation is None:
        st.warning("Please upload and process your PDFs first.")
        return

    st.session_state.display_messages.append(
        {"role": "user", "content": user_question}
    )
    with chat_container:
        render_chat_history()

    try:
        with st.spinner("Generating answer..."):
            response = st.session_state.conversation.invoke({"question": user_question})
    except Exception as exc:
        st.error(f"Chat failed: {exc}")
        return

    st.session_state.chat_history = response.get("chat_history", [])
    st.session_state.last_sources = response.get("source_documents", [])

    answer = response.get("answer", "").strip() or "No answer returned."
    st.session_state.display_messages.append(
        {"role": "assistant", "content": answer}
    )
    st.rerun()


def reset_chat_state(mode: str):
    st.session_state.selected_mode = mode
    st.session_state.conversation = None
    st.session_state.chat_history = []
    st.session_state.last_sources = []
    st.session_state.display_messages = []


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with PDFs", page_icon=":robot_face:")
    st.write(css, unsafe_allow_html=True)

    if "selected_mode" not in st.session_state:
        st.session_state.selected_mode = list(MODE_OPTIONS.values())[0]
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "last_sources" not in st.session_state:
        st.session_state.last_sources = []
    if "display_messages" not in st.session_state:
        st.session_state.display_messages = []

    with st.sidebar:
        st.subheader("Your documents")
        selected_label = st.selectbox("Choose a backend", list(MODE_OPTIONS.keys()))
        selected_mode = MODE_OPTIONS[selected_label]

        if st.session_state.selected_mode != selected_mode:
            reset_chat_state(selected_mode)

        st.caption("Upload one or more PDFs, then click Process.")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here",
            type=["pdf"],
            accept_multiple_files=True,
        )

        if st.button("Process"):
            if not pdf_docs:
                st.warning("Please upload at least one PDF.")
            else:
                try:
                    st.session_state.conversation = None
                    st.session_state.chat_history = []
                    st.session_state.last_sources = []
                    st.session_state.display_messages = []
                    with st.spinner(f"Processing PDFs with {selected_label}..."):
                        process_pdfs(selected_mode, pdf_docs)
                        st.session_state.conversation = build_chain(selected_mode)
                    st.success("Documents processed. You can start asking questions.")
                except Exception as exc:
                    st.session_state.conversation = None
                    st.session_state.chat_history = []
                    st.session_state.last_sources = []
                    st.session_state.display_messages = []
                    st.error(f"Processing failed: {exc}")

        if st.session_state.last_sources:
            st.markdown("**Recent sources**")
            for doc in st.session_state.last_sources[:3]:
                filename = doc.metadata.get("filename", "unknown")
                page_num = doc.metadata.get("page_num", "unknown")
                st.write(f"- {filename}, page {page_num}")

    st.header("Chat with PDFs :robot_face:")
    st.caption(f"Current backend: {selected_label}")

    chat_container = st.container()
    with chat_container:
        if st.session_state.display_messages:
            render_chat_history()

    user_question = st.chat_input("Ask questions about your documents")
    if user_question:
        handle_userinput(user_question, chat_container)


if __name__ == "__main__":
    main()
