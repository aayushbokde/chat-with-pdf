import streamlit as st
from utils import extract_text_from_pdf, split_text_into_chunks, search_index, generate_gemini_answer, summarize_text_with_gemini
from sklearn.feature_extraction.text import TfidfVectorizer

# Set page config
st.set_page_config(page_title='Chat With PDF', layout='centered')
st.title("Chat With your PDF")

# Upload PDF
uploaded_file = st.file_uploader("Upload Your Pdf file", type=["pdf"])

if uploaded_file:
    text = extract_text_from_pdf(uploaded_file)
    chunks = split_text_into_chunks(text)

    # Creating a TF-IDF index
    vectorizer = TfidfVectorizer()
    index = vectorizer.fit_transform(chunks)

    # Ask a question
    question = st.text_input("Ask a Question about the PDF: ")

    if question:
        # Check for summary-style question
        summary_keywords = ["what is this file about", "summarize", "overview", "brief", "summary"]

        if any(kw in question.lower() for kw in summary_keywords):
            full_text = extract_text_from_pdf(uploaded_file)
            answer = summarize_text_with_gemini(full_text)
            top_chunks = []  # No context to show for summaries
        else:
            # Retrieve top relevant chunks
            top_chunks = search_index(question, vectorizer, index, chunks)
            context = " ".join(top_chunks)
            answer = generate_gemini_answer(question, top_chunks)

        # Show final result
        st.subheader("Answer:")
        st.write(answer)

        # Show chunks used (only if not a summary)
        if top_chunks:
            with st.expander("Context Used"):
                for i, chunk in enumerate(top_chunks, 1):
                    st.markdown(f"**{i}.** {chunk}")
