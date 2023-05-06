import os
import json
import boto3
import logging
from typing import List, Callable
from urllib.parse import urlparse
from langchain.vectorstores import FAISS
from langchain.vectorstores import OpenSearchVectorSearch
from langchain.embeddings import SagemakerEndpointEmbeddings
from langchain.llms.sagemaker_endpoint import SagemakerEndpoint
from langchain.llms.sagemaker_endpoint import LLMContentHandler
from langchain.llms.sagemaker_endpoint import ContentHandlerBase
from .fastapi_request import SAGEMAKER_ENDPOINT_MAPPING, Request
from langchain.embeddings.sagemaker_endpoint import EmbeddingsContentHandler

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

class ContentHandlerForEmbeddings(EmbeddingsContentHandler):
    """
    encode input string as utf-8 bytes, read the embeddings
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
            return [embeddings[0]]
        return embeddings

class ContentHandlerForTextGeneration(LLMContentHandler):
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

def _get_credentials(secret_id: str, region_name: str) -> str:

    client = boto3.client('secretsmanager', region_name=region_name)
    response = client.get_secret_value(SecretId=secret_id)
    secrets_value = json.loads(response['SecretString'])
    return secrets_value

def load_vector_db_opensearch(secret_id: str,
                              region: str,
                              opensearch_domain_endpoint: str,
                              opensearch_index: str,
                              embeddings_model: str) -> OpenSearchVectorSearch:
    logger.info(f"load_vector_db_opensearch, secret_id={secret_id}, region={region}, "
                f"opensearch_domain_endpoint={opensearch_domain_endpoint}, opensearch_index={opensearch_index}, "  
                f"embeddings_model={embeddings_model}")
    opensearch_domain_endpoint = f"https://{opensearch_domain_endpoint}"
    embeddings_model_endpoint = SAGEMAKER_ENDPOINT_MAPPING[embeddings_model]
    logger.info(f"embeddings_model_endpoint={embeddings_model_endpoint}, opensearch_domain_endpoint={opensearch_domain_endpoint}")
    creds = _get_credentials(secret_id, region)
    http_auth = (creds['username'], creds['password'])
    vector_db = OpenSearchVectorSearch(index_name=opensearch_index,
                                       embedding_function=_create_sagemaker_embeddings(embeddings_model_endpoint,
                                                                                       region),
                                       opensearch_url=opensearch_domain_endpoint,
                                       http_auth=http_auth)
    logger.info(f"returning handle to OpenSearchVectorSearch, vector_db={vector_db}")
    return vector_db

def load_vector_db_faiss(vectordb_s3_path: str, vectordb_local_path: str, embeddings_endpoint_name: str, region: str) -> FAISS:
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

def setup_sagemaker_endpoint_for_text_generation(req: Request, region: str = "us-east-1") -> Callable:
    parameters = {
    "max_length": req.max_length,
    "num_return_sequences": req.num_return_sequences,
    "top_k": req.top_k,
    "top_p": req.top_p,
    "do_sample": req.do_sample,
    "temperature": req.temperature,}
    
    endpoint_name = req.text_generation_model
    content_handler = ContentHandlerForTextGeneration()    
    print(f"SAGEMAKER_ENDPOINT_MAPPING[{endpoint_name}]={SAGEMAKER_ENDPOINT_MAPPING[endpoint_name]}")
    sm_llm = SagemakerEndpoint(
        endpoint_name=SAGEMAKER_ENDPOINT_MAPPING[endpoint_name],
        region_name=region,
        model_kwargs=parameters,
        content_handler=content_handler)
    return sm_llm

