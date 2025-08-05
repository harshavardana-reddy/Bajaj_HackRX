from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List, Optional
import logging
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import numpy as np
import requests
import PyPDF2
from docx import Document
import email
from email import policy
import io
import json
from typing import Dict, Any

# Initialize FastAPI app
app = FastAPI(
    title="Insurance Policy Query System",
    description="LLM-powered insurance policy analysis system",
    version="1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load environment variables
load_dotenv()
TEAM_TOKEN = os.getenv("TEAM_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")

# Initialize clients
security = HTTPBearer()
embedder = SentenceTransformer('all-mpnet-base-v2')

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)
generation_config = GenerationConfig(
    temperature=0.1,
    top_p=0.95,
    top_k=40,
    max_output_tokens=1024,
)

safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create index if it doesn't exist
if PINECONE_INDEX not in pc.list_indexes().names():
    pc.create_index(
        name=PINECONE_INDEX,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

pinecone_index = pc.Index(PINECONE_INDEX)

class DocumentQueryRequest(BaseModel):
    documents: str  # URL to the policy document
    questions: List[str]

class DocumentQueryResponse(BaseModel):
    answers: List[str]

class TextProcessor:
    """Handles text cleaning and chunking"""
    def __init__(self, chunk_size=1000, overlap=200):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        text = ' '.join(text.split())
        return text.strip()
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks"""
        words = text.split()
        chunks = []
        for i in range(0, len(words), self.chunk_size - self.overlap):
            chunk = ' '.join(words[i:i + self.chunk_size])
            chunks.append(chunk)
        return chunks

class DocumentProcessor:
    """Processes different document types"""
    def __init__(self):
        self.text_processor = TextProcessor()
    
    async def process_document(self, url: str, content_type: str) -> List[dict]:
        """Process document and return chunks with metadata"""
        try:
            response = requests.get(url)
            response.raise_for_status()
            content = response.content
            
            if content_type == "pdf":
                text = self._extract_pdf_text(content)
            elif content_type == "docx":
                text = self._extract_docx_text(content)
            elif content_type == "email":
                text = self._extract_email_text(content)
            else:
                raise ValueError("Unsupported document type")
            
            clean_text = self.text_processor.clean_text(text)
            chunks = self.text_processor.chunk_text(clean_text)
            
            # Generate embeddings for each chunk
            embeddings = embedder.encode(chunks)
            
            # Prepare Pinecone upsert data
            vectors = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                vectors.append({
                    "id": f"{url}-{i}",
                    "values": embedding.tolist(),
                    "metadata": {
                        "text": chunk,
                        "source": url,
                        "chunk_num": i
                    }
                })
            
            # Upsert to Pinecone
            pinecone_index.upsert(vectors=vectors)
            
            return chunks
        
        except Exception as e:
            logging.error(f"Document processing failed: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Document processing failed: {str(e)}")

    def _extract_pdf_text(self, content: bytes) -> str:
        try:
            with io.BytesIO(content) as pdf_file:
                reader = PyPDF2.PdfReader(pdf_file)
                text = "\n".join([page.extract_text() for page in reader.pages])
            return text
        except Exception as e:
            logging.error(f"PDF extraction failed: {str(e)}")
            raise

    def _extract_docx_text(self, content: bytes) -> str:
        try:
            with io.BytesIO(content) as docx_file:
                doc = Document(docx_file)
                text = "\n".join([para.text for para in doc.paragraphs])
            return text
        except Exception as e:
            logging.error(f"DOCX extraction failed: {str(e)}")
            raise

    def _extract_email_text(self, content: bytes) -> str:
        try:
            msg = email.message_from_bytes(content, policy=policy.default)
            text = ""
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        text += part.get_payload()
            else:
                text = msg.get_payload()
            return text
        except Exception as e:
            logging.error(f"Email extraction failed: {str(e)}")
            raise

def validate_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != TEAM_TOKEN:
        raise HTTPException(
            status_code=401,
            detail="Invalid authorization token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials

async def semantic_search(query: str, document_url: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Perform semantic search using Pinecone"""
    try:
        query_embedding = embedder.encode(query).tolist()
        results = pinecone_index.query(
            vector=query_embedding,
            filter={"source": {"$eq": document_url}},
            top_k=top_k,
            include_metadata=True
        )
        return [match.metadata for match in results.matches]
    except Exception as e:
        logging.error(f"Semantic search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

async def generate_answer(question: str, context: List[str]) -> str:
    """Generate answer using Gemini with relevant context"""
    try:
        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        
        context_str = "\n\n".join([f"--- Context Excerpt {i+1} ---\n{text}" for i, text in enumerate(context)])
        
        prompt = f"""You are an expert insurance policy analyst. Your task is to answer questions about insurance policies 
        with extreme accuracy based ONLY on the provided policy excerpts. Follow these rules:

        1. Answer concisely but completely, including all relevant details from the context
        2. If the information is not in the context, respond with: "The policy does not specify."
        3. Never make up information or speculate beyond what's in the context
        4. For numerical values (like waiting periods, limits), always state the exact number from the context
        5. For coverage questions, clearly state what is covered and any conditions or limitations

        Policy Context:
        {context_str}

        Question: {question}

        Answer:"""
        
        response = model.generate_content(prompt)
        
        if response.candidates and response.candidates[0].content.parts:
            answer = response.candidates[0].content.parts[0].text
            # Post-processing to ensure accuracy
            if "not in the context" in answer.lower() or "not specified" in answer.lower():
                return "The policy does not specify."
            return answer.strip()
        return "The policy does not specify."
    except Exception as e:
        logging.error(f"Gemini API error: {str(e)}")
        return f"Error generating answer: {str(e)}"

@app.post("/hackrx/run", response_model=DocumentQueryResponse)
async def process_policy_queries(
    request: DocumentQueryRequest,
    token: str = Depends(validate_token)
):
    """
    Process policy document and answer questions using LLM-powered search
    """
    try:
        # Initialize processors
        doc_processor = DocumentProcessor()
        
        # Process document (extract text, chunk, and index)
        await doc_processor.process_document(request.documents, "pdf")
        
        # Answer each question
        answers = []
        for question in request.questions:
            try:
                # 1. Semantic search for relevant chunks
                relevant_chunks = await semantic_search(question, request.documents)
                context = [chunk["text"] for chunk in relevant_chunks]
                
                if not context:
                    answers.append("The policy does not specify.")
                    continue
                
                # 2. Generate LLM answer
                answer = await generate_answer(question, context)
                answers.append(answer)
                
            except Exception as e:
                logging.error(f"Error processing question '{question}': {str(e)}")
                answers.append(f"Error processing question: {question}")
        
        return DocumentQueryResponse(answers=answers)
    
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Vercel handler
from mangum import Adapter

# Create handler for Vercel
handler = Adapter(app)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=8000)