import os
from enum import Enum
from pydantic import BaseModel

class Text2TextModelName(str, Enum):
    flant5xxl = "flan-t5-xxl"

class EmbeddingsModelName(str, Enum):
    gptj6b = "gpt-j-6b"

class VectorDBType(str, Enum):
    OPENSEARCH = "opensearch"
    FAISS = "faiss"

class Request(BaseModel):
    q: str    
    max_length: int = 500
    num_return_sequences: int = 1
    top_k: int = 50
    top_p: float = 0.95
    do_sample: bool = False
    verbose: bool = False
    max_matching_docs: int = 3  
    text_generation_model: Text2TextModelName = Text2TextModelName.flant5xxl
    embeddings_generation_model: EmbeddingsModelName = EmbeddingsModelName.gptj6b
    vectordb_s3_path: str = None
    vectordb_type: VectorDBType = VectorDBType.OPENSEARCH

SAGEMAKER_ENDPOINT_MAPPING = {
    Text2TextModelName.flant5xxl: os.environ.get('TEXT2TEXT_ENDPOINT_NAME'),
    EmbeddingsModelName.gptj6b: os.environ.get('EMBEDDING_ENDPOINT_NAME'),
}
