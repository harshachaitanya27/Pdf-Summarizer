import streamlit as st
from pdf2image import convert_from_bytes
from crewai import Agent, LLM, Task, Crew
from paddleocr import PaddleOCR
import numpy as np
import concurrent.futures
import threading


st.title("PDF OCR & Summarization App")

# Set up the LLM once (consider moving this to a configuration file in production)
llm = LLM(
    model="ollama/llama3:latest",
    base_url="http://localhost:11434"
)

# Allow multiple PDF uploads
uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

# Initialize PaddleOCR once (this loads the model and should be reused)
ocr = PaddleOCR(use_angle_cls=True, lang="en")

# Create a global lock for accessing the OCR instance
ocr_lock = threading.Lock()

def process_page(image, page_num):
    """
    Process a single page: convert the image to a numpy array,
    run OCR (using a lock to prevent concurrent access), and return the extracted text.
    """
    img_array = np.array(image)
    with ocr_lock:
        result = ocr.ocr(img_array, cls=True)
    page_text = ""
    for line in result:
        words = [word_info[1][0] for word_info in line]
        page_text += " ".join(words) + "\n"
    return page_num, page_text

def summarize_text(agent, text):
    """
    Uses the provided Crew.AI summarization agent to summarize the page text.
    """
    text_summary_task = Task(
        description=f'''Understand the following text and summarize it in 1-2 paragraphs:
{text}, make sure that each paragraph is no more than 100 words''',
        agent=agent,
        expected_output='''Summary:
(first paragraph of summarized text)

(second paragraph of summarized text)
'''
    )

    crew = Crew(
        agents=[agent],
        tasks=[text_summary_task],
    )
    
    crew_output = crew.kickoff(inputs={'text': text})
    return crew_output.raw

if uploaded_files:
    for uploaded_file in uploaded_files:
        st.header(f"Processing file: {uploaded_file.name}")
        file_bytes = uploaded_file.read()
        
        try:
            # Convert PDF pages to images (ensure poppler is installed and in PATH)
            images = convert_from_bytes(file_bytes)
        except Exception as e:
            st.error(f"Error converting PDF: {e}")
            continue

        full_text = ""
        page_texts = {}

        # Process pages concurrently for OCR
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_page, image, i) for i, image in enumerate(images, start=1)]
            for future in concurrent.futures.as_completed(futures):
                page_num, page_text = future.result()
                page_texts[page_num] = page_text
                full_text += f"Page {page_num}:\n{page_text}\n"

        # Display the full extracted text
        st.subheader("Full Extracted Text")
        st.text_area("OCR Output", full_text, height=300)
        
        # Provide a download button for the extracted text
        download_filename = uploaded_file.name.replace(".pdf", ".txt")
        st.download_button("Download Extracted Text", full_text, file_name=download_filename)

        # Create a single summarization agent per file to reduce overhead
        summarizer_agent = Agent(
            name='Text Summarizer Agent',
            role='Content Writer and Text Summarizer',
            llm=llm,
            goal='''You need to understand the text you receive and summarize it in 1-2 paragraphs,
capturing the essence of the content without adding any external information.
Ensure that each paragraph is concise (no more than 100 words).''',
            backstory='''You are an experienced English teacher and content writer with exceptional skills in summarizing text.''',
            memory=True,
            verbose=True,
        )
        
        st.subheader("Page Summaries")
        # Optionally, you can also process summarization concurrently if your API supports it.
        for page_num in sorted(page_texts.keys()):
            page_text = page_texts[page_num]
            summary = summarize_text(summarizer_agent, page_text)
            st.markdown(f"**Page {page_num} Summary:**")
            st.write(summary)