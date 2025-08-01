from src.vector_store import VectorStoreBuilder
from src.recommender import Recommender
from config.config import GROQ_API,MODEL_NAME
from utils.logger import get_logger
from utils.custom_exceptions import CustomException
import os

logger = get_logger(__name__)

class AnimationRecommenderPipeline:
    def __init__(self, persist_dir = "chroma_db"):
         try: 
            logger.info("Initializing Animation Recommender Pipeline...")
            csv_path = os.path.join(os.getcwd(), "data", "anime_with_synopsis.csv")
            self.vector_store = VectorStoreBuilder(csv_path=csv_path,persist_dir=persist_dir)
            retriever = self.vector_store.load_vector_store().as_retriever()
            if GROQ_API is None:
                logger.error("GROQ_API key is not set.")
                raise CustomException("GROQ_API key must be provided and cannot be None.")
            self.recommender = Recommender(
                retriever= retriever,
                api_key=GROQ_API,
                model_name=MODEL_NAME
            )
            logger.info("Animation Recommender Pipeline initialized successfully.")
         except Exception as e:
            logger.error(f"Error initializing AnimationRecommenderPipeline: {e}")
            raise CustomException("Failed to initialize the recommender pipeline.")    
        
    def recommend(self, query):
        try:
            logger.info(f"received a  query: {query}")
            result = self.recommender.recommend(query)
            logger.info("Recommendation processed successfully.")
            return result
        except Exception as e:
            logger.error(f"Error processing recommendation query: {e}")
            raise CustomException("Failed to process the recommendation query.")

    