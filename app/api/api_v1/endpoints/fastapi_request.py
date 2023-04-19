from enum import Enum
from pydantic import BaseModel

class Text2TextModelName(str, Enum):
    flant5xxl = "flan-t5-xxl"

class EmbeddingsModelName(str, Enum):
    gptj6b = "gpt-j-6b"

class Request(BaseModel):
    q: str    
    max_length: int = 500
    num_return_sequences: int = 1
    top_k: int = 50
    top_p: float = 0.95
    do_sample: bool = False
    verbose: bool = False
    max_matching_docs: int = 3  
    text_generation_endpoint: Text2TextModelName = Text2TextModelName.flant5xxl
    embeddings_generation_endpoint: EmbeddingsModelName = EmbeddingsModelName.gptj6b
    endpoint_region: str = "us-east-1"
    vectordb_s3_path: str = "s3://sagemaker-us-east-1-015469603702/qa-w-rag/vectordb/"

SAGEMAKER_ENDPOINT_MAPPING = {
    Text2TextModelName.flant5xxl: "qa-w-rag-huggingface-text2text-flan-t5--2023-04-15-13-25-17-476",
    EmbeddingsModelName.gptj6b: "qa-w-rag-huggingface-textembedding-gpt--2023-04-15-13-32-22-769"
}
