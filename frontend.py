import streamlit as st
from backend import process_video, ask_question

# Page configuration
st.set_page_config(
    page_title="AskTheVideo",
    page_icon="",
    layout="wide"
)

st.title("Ask The Video")
st.markdown("Get answers from YouTube videos using AI-powered Q&A")

# Initialize session state
if 'main_chain' not in st.session_state:
    st.session_state.main_chain = None
if 'video_processed' not in st.session_state:
    st.session_state.video_processed = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Helper function
def is_valid_url(url):
    # Basic validation for YouTube URL
    return "youtube.com/watch" in url or "youtu.be/" in url


# Sidebar for video processing
with st.sidebar:
    st.header("📹 Video Setup")

    video_url = st.text_input(
        "Enter YouTube Video URL:",
        placeholder="https://www.youtube.com/watch?v=..."
    )

    process_button = st.button(
        "Process Video", type="primary", use_container_width=True)

    if process_button:
        if not video_url:
            st.error("Please enter a YouTube URL")
        elif not is_valid_url(video_url):
            st.error("Please enter a valid YouTube URL")
        else:
            with st.spinner("Processing video... This may take a minute."):
                try:
                    # Process the video and create the chain
                    st.session_state.main_chain = process_video(video_url)
                    st.session_state.video_processed = True
                    st.session_state.chat_history = []  # Reset chat history
                    st.success("✅ Video processed successfully! You can now ask questions.")
                except Exception as e:
                    st.error(f"Error processing video: {str(e)}")
                    st.session_state.video_processed = False

    # Display status
    st.divider()
    if st.session_state.video_processed:
        st.success("🟢 Video Ready")
    else:
        st.info("⚪ No video processed yet")

    # Clear button
    if st.button("Clear Session", use_container_width=True):
        st.session_state.main_chain = None
        st.session_state.video_processed = False
        st.session_state.chat_history = []
        st.rerun()

# Main area for Q&A
st.header("💬 Ask Questions")

if not st.session_state.video_processed:
    st.info("👈 Please process a YouTube video from the sidebar to get started.")
else:
    # Display chat history
    for i, chat in enumerate(st.session_state.chat_history):
        with st.chat_message("user"):
            st.write(chat["question"])
        with st.chat_message("assistant"):
            st.write(chat["answer"])

    # Question input
    question = st.chat_input("Ask a question about the video...")

    if question:
        # Display user question
        with st.chat_message("user"):
            st.write(question)

        # Get and display answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer = ask_question(st.session_state.main_chain, question)
                    st.write(answer)

                    # Save to chat history
                    st.session_state.chat_history.append({
                        "question": question,
                        "answer": answer
                    })
                except Exception as e:
                    st.error(f"Error getting answer: {str(e)}")

# Footer
st.divider()
st.caption("© 2026 Mohd Faraz Akram | Built with Streamlit and LangChain")