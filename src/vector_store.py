from langchain.text_splitter import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import CSVLoader
from langchain_huggingface import HuggingFaceEmbeddings

from dotenv import load_dotenv
load_dotenv()

class VectorStoreBuilder:
    def __init__(self,csv_path:str,persist_dir:str = "chroma_db"):
        self.csv_path = csv_path
        self.persist_dir = persist_dir
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
       
    def build_and_save_vector_store(self):
        # Load CSV data
    
        loader = CSVLoader(file_path=self.csv_path, encoding='utf-8',)
        documents = loader.load()

        # Split text into chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        split_docs = text_splitter.split_documents(documents)

        # Create and persist the vector store
        vector_store = Chroma.from_documents(
            documents=split_docs,
            embedding=self.embeddings,
            persist_directory=self.persist_dir
        )
        
        vector_store.persist()
        return vector_store
    
    def load_vector_store(self):
        # Load the persisted vector store
        vector_store = Chroma(
            persist_directory=self.persist_dir,
            embedding_function=self.embeddings
        )
        return vector_store

