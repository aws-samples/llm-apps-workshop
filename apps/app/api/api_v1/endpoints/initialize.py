import os
import json
import boto3
import logging
from typing import List, Callable
from urllib.parse import urlparse
from langchain.vectorstores import FAISS
from .fastapi_request import SAGEMAKER_ENDPOINT_MAPPING
from langchain.embeddings import SagemakerEndpointEmbeddings
from langchain.llms.sagemaker_endpoint import SagemakerEndpoint
from langchain.llms.sagemaker_endpoint import ContentHandlerBase

logger = logging.getLogger(__name__)
class SagemakerEndpointEmbeddingsJumpStart(SagemakerEndpointEmbeddings):
    def embed_documents(
        self, texts: List[str], chunk_size: int = 5
    ) -> List[List[float]]:
        """Compute doc embeddings using a SageMaker Inference Endpoint.

        Args:
            texts: The list of texts to embed.
            chunk_size: The chunk size defines how many input texts will
                be grouped together as request. If None, will use the
                chunk size specified by the class.

        Returns:
            List of embeddings, one for each text.
        """
        results = []
        #print(f"length of texts = {len(texts)}")
        _chunk_size = len(texts) if chunk_size > len(texts) else chunk_size
        
        for i in range(0, len(texts), _chunk_size):
            response = self._embedding_func(texts[i : i + _chunk_size])
            #print(response)
            results.extend(response)
        return results

class ContentHandlerForEmbeddings(ContentHandlerBase):
    """
    encode input string as uf-8 bytes, read the embeddings
    from the output
    """
    content_type = "application/json"
    accepts = "application/json"
    def transform_input(self, prompt: str, model_kwargs = {}) -> bytes:
        input_str = json.dumps({"text_inputs": prompt, **model_kwargs})
        return input_str.encode('utf-8') 
    
    def transform_output(self, output: bytes) -> str:
        response_json = json.loads(output.read().decode("utf-8"))
        embeddings = response_json["embedding"]
        if len(embeddings) == 1:
            return embeddings[0]
        return embeddings

class ContentHandlerForTextGeneration(ContentHandlerBase):
    content_type = "application/json"
    accepts = "application/json"
    
    def transform_input(self, prompt: str, model_kwargs = {}) -> bytes:
        input_str = json.dumps({"text_inputs": prompt, **model_kwargs})
        return input_str.encode('utf-8')
    
    def transform_output(self, output: bytes) -> str:
        response_json = json.loads(output.read().decode("utf-8"))
        return response_json["generated_texts"][0]

def _create_sagemaker_embeddings(endpoint_name: str, region: str = "us-east-1") -> SagemakerEndpointEmbeddingsJumpStart:
    # create a content handler object which knows how to serialize
    # and deserialize communication with the model endpoint
    content_handler = ContentHandlerForEmbeddings()

    # read to create the Sagemaker embeddings, we are providing
    # the Sagemaker endpoint that will be used for generating the
    # embeddings to the class
    embeddings = SagemakerEndpointEmbeddingsJumpStart( 
        endpoint_name=endpoint_name,
        region_name=region, 
        content_handler=content_handler
    )
    logger.info(f"embeddings type={type(embeddings)}")

    return embeddings

def load_vector_db(vectordb_s3_path: str, vectordb_local_path: str, embeddings_endpoint_name: str, region: str) -> FAISS:
    os.makedirs(vectordb_local_path, exist_ok=True)
    # download the vectordb files from S3
    # note that the following code is only applicable to FAISS
    # would need to be enhanced to support other vector dbs
    vectordb_files = ["index.pkl", "index.faiss"]
    for vdb_file in vectordb_files:        
        s3 = boto3.client('s3')
        fpath = os.path.join(vectordb_local_path, vdb_file)
        with open(fpath, 'wb') as f:
            parsed = urlparse(vectordb_s3_path)
            bucket = parsed.netloc
            path =  os.path.join(parsed.path[1:], vdb_file)
            logger.info(f"going to download from bucket={bucket}, path={path}, to {fpath}")
            s3.download_fileobj(bucket, path, f)
            logger.info(f"after downloading from bucket={bucket}, path={path}, to {fpath}")

    # files are downloaded, lets load the vectordb
    logger.info("creating a Sagemaker embeddings object to hydrate the vector db")
    embeddings = _create_sagemaker_embeddings(SAGEMAKER_ENDPOINT_MAPPING[embeddings_endpoint_name], region)
    vector_db = FAISS.load_local(vectordb_local_path, embeddings)
    logger.info(f"vector db hydrated, type={type(vector_db)} it has {vector_db.index.ntotal} embeddings")

    return vector_db

def setup_sagemaker_endpoint_for_text_generation(endpoint_name: str, region: str = "us-east-1") -> Callable:
    parameters = {
    "max_length": 200,
    "num_return_sequences": 1,
    "top_k": 250,
    "top_p": 0.95,
    "do_sample": False,
    "temperature": 1,}

    content_handler = ContentHandlerForTextGeneration()    
    print(f"AGEMAKER_ENDPOINT_MAPPING[endpoint_name]={SAGEMAKER_ENDPOINT_MAPPING[endpoint_name]}")
    sm_llm = SagemakerEndpoint(
        endpoint_name=SAGEMAKER_ENDPOINT_MAPPING[endpoint_name],
        region_name=region,
        model_kwargs=parameters,
        content_handler=content_handler)
    return sm_llm

