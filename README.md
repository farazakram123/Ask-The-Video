# AskTheVideo 🎥
[Live Link of Application](https://ask-the-video.streamlit.app/)

An AI-powered Q&A application that allows you to ask questions about YouTube videos. The app fetches video transcripts, processes them using RAG (Retrieval Augmented Generation), and provides intelligent answers based on the video content.

## Features

- 🎯 **YouTube Video Processing**: Automatically fetches and processes YouTube video transcripts
- 💬 **Intelligent Q&A**: Ask questions and get accurate answers from video content
- 🔍 **Semantic Search**: Uses vector embeddings for relevant context retrieval
- 📝 **Chat History**: Maintains conversation history during your session
- 🎨 **User-Friendly Interface**: Clean Streamlit interface for easy interaction

## Tech Stack

- **Frontend**: Streamlit
- **Backend**: Python
- **LLM Framework**: LangChain
- **Vector Database**: Pinecone
- **Embeddings**: HuggingFace (sentence-transformers/all-MiniLM-L6-v2)
- **Language Model**: HuggingFace (openai/gpt-oss-20b)
- **Transcript API**: Supadata API

## Prerequisites

- Python 3.8+
- API Keys for:
  - Pinecone
  - HuggingFace
  - Supadata

## Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd AskTheVideo
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**
   - Windows:
     ```bash
     venv\Scripts\activate
     ```
   - macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

Create a `.env` file in the project root with the following variables:

```env
PINECONE_API_KEY=your_pinecone_api_key
INDEX_NAME=video-processing
HF_TOKEN=your_huggingface_token
SUPADATA_API_KEY=your_supadata_api_key
```

### Getting API Keys

- **Pinecone**: Sign up at [pinecone.io](https://www.pinecone.io/)
- **HuggingFace**: Get your token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
- **Supadata**: Register at [supadata.ai](https://supadata.ai/)

## Usage

1. **Start the application**
   ```bash
   streamlit run frontend.py
   ```

2. **Process a YouTube video**
   - Enter a YouTube URL in the sidebar
   - Click "Process Video"
   - Wait for the video to be processed

3. **Ask questions**
   - Once processed, type your questions in the chat input
   - Get answers based on the video content

## Project Structure

```
AskTheVideo/
│
├── backend.py          # Core logic: transcript fetching, RAG chain setup
├── frontend.py         # Streamlit UI and user interaction
├── .env               # Environment variables (not in version control)
├── .gitignore         # Git ignore file
└── README.md          # Project documentation
```

## How It Works

1. **Transcript Extraction**: Fetches YouTube video transcript using Supadata API
2. **Text Chunking**: Splits transcript into manageable chunks with overlap
3. **Embedding Generation**: Converts chunks into vector embeddings using HuggingFace
4. **Vector Storage**: Stores embeddings in Pinecone for semantic search
5. **Question Processing**: User questions are embedded and matched with relevant chunks
6. **Answer Generation**: LLM generates answers based on retrieved context

## Features in Detail

### Video Processing
- Validates YouTube URLs
- Extracts video ID automatically
- Handles different YouTube URL formats (watch?v= and youtu.be/)
- Processes transcripts with error handling

### Q&A System
- Uses similarity search to find relevant transcript sections
- Returns top 3 most relevant chunks
- Generates contextual answers using LLM
- Falls back gracefully when information isn't available

### Session Management
- Maintains chat history within a session
- Clear session functionality to start fresh
- Visual status indicators for video processing state

## Troubleshooting

- **No transcript available**: Some videos may not have transcripts enabled
- **API rate limits**: Check your API usage if requests fail
- **Memory issues**: Processing very long videos may require more resources

## Contributing

Contributions are welcome! Feel free to submit issues or pull requests.

## License

© 2026 Mohd Faraz Akram

## Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Powered by [LangChain](https://www.langchain.com/)
- Vector storage by [Pinecone](https://www.pinecone.io/)
- Embeddings from [HuggingFace](https://huggingface.co/)
- Transcript API by [Supadata](https://supadata.ai/)
