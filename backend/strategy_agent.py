from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import os
import requests
import xml.etree.ElementTree as ET
from PyPDF2 import PdfReader
from pdfminer.high_level import extract_text as pdfminer_extract_text
import docx
from io import BytesIO
from PIL import Image
import pytesseract
from pdf2image import convert_from_bytes
from dotenv import load_dotenv
from pinecone import Pinecone
from openai import OpenAI
from datetime import datetime
import re

# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

pc = Pinecone(api_key=PINECONE_API_KEY)
pinecone_index = pc.Index(PINECONE_INDEX_NAME)
client = OpenAI(api_key=OPENAI_API_KEY)
EMBEDDING_MODEL = "text-embedding-ada-002"

app = FastAPI()

class InputText(BaseModel):
    text: str

def get_embedding(text):
    response = client.embeddings.create(
        input=text,
        model=EMBEDDING_MODEL
    )
    embedding = response.data[0].embedding
    return embedding

def similarity_search(embedding, top_k=3):
    result = pinecone_index.query(vector=embedding, top_k=top_k, include_metadata=True)
    return result

def generate_strategy(context_metadata):
    prompt = f"""
    You are a legal strategy generation agent. Based on the following similar court case metadata, provide a strategy:
    
    Facts: {context_metadata.get('Facts', 'N/A')}
    Issues: {context_metadata.get('Issues', 'N/A')}
    Reasoning: {context_metadata.get('Reasoning', 'N/A')}
    Decision: {context_metadata.get('Decision', 'N/A')}
    
    Please propose a comprehensive legal strategy based on the above metadata:
    """

    gpt_response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful AI assistant. Please summarize the legal case into the specified categories."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=1500,
        temperature=0.5
    )
    return gpt_response.choices[0].message.content

def fetch_arxiv_papers(query: str, max_results: int = 5):
    base_url = "http://export.arxiv.org/api/query"
    params = {
        "search_query": query,
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending"
    }

    response = requests.get(base_url, params=params)
    if response.status_code != 200:
        return f"Error fetching papers: {response.status_code}"
    
    root = ET.fromstring(response.content)
    papers = []
    for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
        title = entry.find("{http://www.w3.org/2005/Atom}title").text.strip()
        summary = entry.find("{http://www.w3.org/2005/Atom}summary").text.strip()
        link = entry.find("{http://www.w3.org/2005/Atom}id").text.strip()
        papers.append((title, summary, link))
    
    if not papers:
        return "No ArXiv papers found for this query."

    return papers

def extract_text_from_pdf(file_bytes):
    """
    Extract text from a PDF using PyPDF2 and pdfminer.six as fallback.
    """
    text_segments = []
    
    # First attempt to extract text using PyPDF2
    try:
        pdf_reader = PdfReader(BytesIO(file_bytes))
        for page in pdf_reader.pages:
            text_segments.append(page.extract_text() or "")
    except Exception as e:
        print(f"PyPDF2 extraction failed: {e}")

    # If PyPDF2 fails or extracts no text, fallback to pdfminer.six
    if not any(text_segments):
        try:
            pdfminer_text = pdfminer_extract_text(BytesIO(file_bytes))
            text_segments.append(pdfminer_text)
        except Exception as e:
            print(f"pdfminer.six extraction failed: {e}")
            raise HTTPException(status_code=500, detail="Failed to extract text from PDF.")
    
    full_text = "\n".join([seg for seg in text_segments if seg.strip()])
    return full_text.strip()

def extract_text_from_file(filename: str, file_bytes: bytes):
    filename = filename.lower()
    if filename.endswith('.txt'):
        return file_bytes.decode("utf-8")
    elif filename.endswith('.pdf'):
        return extract_text_from_pdf(file_bytes)
    elif filename.endswith('.docx'):
        doc = docx.Document(BytesIO(file_bytes))
        full_text = [para.text for para in doc.paragraphs]
        return "\n".join(full_text)
    elif filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
        image = Image.open(BytesIO(file_bytes))
        text = pytesseract.image_to_string(image)
        return text
    else:
        return None
    
@app.post("/process_file")
async def process_file(file: UploadFile = File(...)):
    file_bytes = await file.read()
    content = extract_text_from_file(file.filename, file_bytes)
    if not content:
        raise HTTPException(status_code=400, detail="Unsupported file format or failed to extract text.")
    embedding = get_embedding(content)
    search_results = similarity_search(embedding, top_k=1)
    if search_results and len(search_results['matches']) > 0:
        best_match = search_results['matches'][0]
        metadata = best_match['metadata']
        strategy = generate_strategy(metadata)
        issues_query = metadata.get('Issues', 'N/A')
        papers = []
        if issues_query != 'N/A':
            papers = fetch_arxiv_papers(issues_query, max_results=3)
        return {"strategy": strategy, "papers": papers}
    else:
        return {"strategy": None, "papers": None, "message": "No similar cases found"}

@app.post("/process_text")
async def process_text(input_text: InputText):
    embedding = get_embedding(input_text.text)
    search_results = similarity_search(embedding, top_k=1)
    if search_results and len(search_results['matches']) > 0:
        best_match = search_results['matches'][0]
        metadata = best_match['metadata']
        strategy = generate_strategy(metadata)
        issues_query = metadata.get('Issues', 'N/A')
        if issues_query == 'N/A':
            match = re.search(r"Issues:\s*(.*?)(?=\n|$)", input_text.text, re.IGNORECASE)
            if match:
                issues_query = match.group(1).strip() or 'N/A'

        papers = []
        if issues_query != 'N/A' and issues_query.strip():
            papers = fetch_arxiv_papers(issues_query, max_results=3)
        return {"strategy": strategy, "papers": papers}
    else:
        match = re.search(r"Issues:\s*(.*?)(?=\n|$)", input_text.text, re.IGNORECASE)
        issues_query = match.group(1).strip() if match else ''
        papers = []
        if issues_query.strip():
            papers = fetch_arxiv_papers(issues_query, max_results=3)
        return {"strategy": None, "papers": papers, "message": "No similar cases found"}
