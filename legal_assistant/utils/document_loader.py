from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import os
import glob
import re

class DocumentLoader:
    def __init__(self, country: str):
        self.country = country
        self.docs_path = f"./data/legal_docs/{country}/"
        #Split document into chunks
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
            add_start_index=True,
        )
    
    def _clean_text(self, text: str) -> str:
        """Clean raw text extracted from PDFs."""
        # Remove multiple newlines
        text = re.sub(r'\n\s*\n', '\n\n', text)
        # Remove non-ASCII characters
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)
        # Replace multiple spaces with a single space
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def load_documents(self):
        """Load, clean, and split legal documents for the country."""
        documents = []
        
        if not os.path.exists(self.docs_path):
            os.makedirs(self.docs_path, exist_ok=True)
            self._create_sample_documents()
        
        
        pdf_files = glob.glob(os.path.join(self.docs_path, "*.pdf"))
        for pdf_file in pdf_files:
            try:
                loader = PyMuPDFLoader(pdf_file)
                docs = loader.load()
                
                for doc in docs:
                    doc.page_content = self._clean_text(doc.page_content) #Built in text formatting function
                
                documents.extend(docs)
                print(f"Loaded and Cleaned PDF: {pdf_file}")
            except Exception as e:
                print(f"Error processing PDF {pdf_file}: {e}")
        
        # Load text files
        txt_files = glob.glob(os.path.join(self.docs_path, "*.txt"))
        for txt_file in txt_files:
            try:
                loader = TextLoader(txt_file, encoding='utf-8')
                docs = loader.load()
                documents.extend(docs)
                print(f"Loaded TXT: {txt_file}")
            except Exception as e:
                print(f"Error loading TXT {txt_file}: {e}")
        
       
        if documents:
            split_docs = self.text_splitter.split_documents(documents)
            print(f"Split {len(documents)} source pages into {len(split_docs)} chunks.")
            return split_docs
        
        return []
    
    def _create_sample_documents(self):
        """Create sample legal documents for demonstration"""
        sample_content = {
            "india": """
Indian Legal System Comprehensive Overview

CONSTITUTIONAL LAW:
The Constitution of India, adopted on 26th January 1950, is the supreme law of the land.

Fundamental Rights (Articles 12-35):
- Right to Equality (Articles 14-18)
- Right to Freedom (Articles 19-22)
- Right against Exploitation (Articles 23-24)
- Right to Freedom of Religion (Articles 25-28)
- Cultural and Educational Rights (Articles 29-30)
- Right to Constitutional Remedies (Article 32)

Directive Principles of State Policy (Articles 36-51):
These are guidelines for the government to establish social and economic democracy.
            """,
            
            "usa": """
United States Legal System Comprehensive Overview

CONSTITUTIONAL LAW:
The U.S. Constitution, ratified in 1788, establishes the framework of federal government.

Bill of Rights (First 10 Amendments):
- First Amendment: Freedom of speech, religion, press
- Fourth Amendment: Search and seizure protections
- Fifth Amendment: Due process and self-incrimination
- Sixth Amendment: Right to counsel and speedy trial
            """,
            
            "germany": """
German Legal System Comprehensive Overview

CONSTITUTIONAL LAW:
The Basic Law (Grundgesetz) of 1949 serves as Germany's constitution.

Fundamental Rights (Articles 1-19):
- Human dignity (Article 1)
- Personal freedom and equality
- Freedom of expression and assembly
- Religious freedom and conscience
            """
        }
        
        # Create sample document file
        sample_file = os.path.join(self.docs_path, f"{self.country}_legal_overview.txt")
        with open(sample_file, 'w', encoding='utf-8') as f:
            f.write(sample_content.get(self.country, "Legal information not available."))
        
        print(f"Created sample legal document for {self.country}")
