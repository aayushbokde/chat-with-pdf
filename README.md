# 📄 Chat With Your PDF

A simple Streamlit web app that allows users to upload a PDF, ask questions about its content, and receive AI-generated answers using Google's Gemini API.

---

## 🔍 Features

- 🧠 **Ask Questions**: Upload any PDF and ask questions in natural language.
- 🤖 **AI-Powered Answers**: Uses **Google Gemini 1.5 Flash** to provide context-aware responses.
- ✂️ **Chunked Context**: Splits PDF content into manageable parts for better processing.
- 📝 **Smart Summary**: Just ask for a "summary" or "overview" to get a brief of the entire PDF.
- ✅ **Secure API**: Loads your Gemini API key from a `.env` file (not committed to GitHub).

---

## 📦 Setup Instructions

1. **Create and activate a virtual environment**
   ```bash
   python -m venv venv
   venv\Scripts\activate   # On Windows
   source venv/bin/activate  # On Mac/Linux
