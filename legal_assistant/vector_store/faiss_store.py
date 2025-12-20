import os
import logging
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from utils.document_loader import DocumentLoader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FAISSVectorStore:
    """Manages FAISS vector stores for different countries."""
    def __init__(self, country: str):
        self.country = country
        # FAISS stores its index in a directory
        self.index_path = f"./faiss_db/{country}"
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Initialize FAISS vector store
        self.vectorstore = self._initialize_vectorstore()
    
    def _initialize_vectorstore(self):
        """Initialize or load an existing FAISS vectorstore."""
        
        # Check if a pre-built FAISS index exists
        if os.path.exists(os.path.join(self.index_path, "index.faiss")):
            logger.info(f"Loading existing FAISS index for {self.country} from {self.index_path}")
            try:
                
                return FAISS.load_local(
                    self.index_path, 
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
            except Exception as e:
                logger.error(f"Failed to load FAISS index for {self.country}: {e}")
                # Fallback to creating a new index if loading fails
                return self._create_new_faiss_index()
        else:
            return self._create_new_faiss_index()

    def _create_new_faiss_index(self):
        """Create a new FAISS index from source documents."""
        logger.info(f"Creating new FAISS index for {self.country}")
        os.makedirs(self.index_path, exist_ok=True)
        
        # Load documents first
        doc_loader = DocumentLoader(self.country)
        documents = doc_loader.load_documents()

        if not documents:
            logger.warning(f"No documents found for {self.country}. Creating an empty FAISS index.")
            # FAISS cannot be empty, so we create it with a dummy document.
            dummy_doc = [Document(page_content="This is a placeholder document to initialize the empty legal database.")]
            vectorstore = FAISS.from_documents(dummy_doc, self.embeddings)
        else:
            logger.info(f"Creating FAISS index with {len(documents)} document chunks for {self.country}")
            vectorstore = FAISS.from_documents(documents, self.embeddings)
        
        # Save the newly created index
        vectorstore.save_local(self.index_path)
        logger.info(f"Successfully saved new FAISS index for {self.country} to {self.index_path}")
        return vectorstore

    def add_documents(self, documents: list[Document]):
        """Add new documents to the existing vectorstore."""
        if not self.vectorstore:
            logger.error("Vectorstore is not initialized. Cannot add documents.")
            return
        
        logger.info(f"Adding {len(documents)} new documents to the FAISS index for {self.country}.")
        self.vectorstore.add_documents(documents)
        # Re-save the updated index to disk
        self.vectorstore.save_local(self.index_path)
        logger.info("Successfully updated and saved the FAISS index.")

