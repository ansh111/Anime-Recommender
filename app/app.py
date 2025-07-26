import os 
import sys
# Add root directory to Python path so pipeline can be imported

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import streamlit as st
from pipeline.pipeline import AnimationRecommenderPipeline
from dotenv import load_dotenv

st.set_page_config(page_title="Anime Recommnder",layout="wide")

load_dotenv()
@st.cache_resource
def get_pipeline():
    return AnimationRecommenderPipeline()

pipeline = get_pipeline()
st.title("Anime Recommender System")

query = st.text_input("Enter your anime prefernces eg. : light hearted anime with school settings")
if query:
    with st.spinner("Fetching recommendations..."):
        try:
            result = pipeline.recommend(query)
            st.success("Recommendations generated successfully!")
            st.markdown("### Recommendations")
            st.write(result)
        except Exception as e:
            st.error(f"An error occurred: {e}")
