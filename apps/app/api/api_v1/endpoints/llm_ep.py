import os
import sys
import logging
from typing import Any, Dict
from fastapi import APIRouter
from urllib.parse import urlparse
from langchain import PromptTemplate
from .fastapi_request import (Request, Text2TextModelName, EmbeddingsModelName)
from .sm_helper import query_sm_endpoint
from langchain.chains.question_answering import load_qa_chain
from .initialize import (setup_sagemaker_endpoint_for_text_generation,
                        load_vector_db)

logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger()
#logging.basicConfig(format='%(asctime)s,%(module)s,%(processName)s,%(levelname)s,%(message)s', level=logging.INFO)

# initialize the vector db as a global variable so that it
# can persist across lambda invocations
VECTOR_DB_DIR = os.path.join("/tmp", "_vectordb")
_vector_db = None
_sm_llm = None

router = APIRouter()

def _init(req: Request):
    # vector db is a global static variable, so that it only
    # created once across multiple lambda invocations, if possible
    global _vector_db
    if _vector_db is None:
        logger.info(f"vector db does not exist, creating it now")
        _vector_db = load_vector_db(req.vectordb_s3_path,
                                    VECTOR_DB_DIR,
                                    req.embeddings_generation_endpoint,
                                    req.endpoint_region)
        logger.info("after creating vector db")
    else:
        logger.info(f"seems like vector db already exists...")

    # just like the vector db the sagemaker endpoint used for
    # text generation is also global and shared across invocations
    # if possible
    global _sm_llm
    if _sm_llm is None:
        logger.info(f"SM LLM endpoint is not setup, setting it up now")
        _sm_llm = setup_sagemaker_endpoint_for_text_generation(req.text_generation_endpoint,
                                                               req.endpoint_region)
        logger.info("after setting up sagemaker llm endpoint")
    else:
        logger.info(f"sagemaker llm endpoint already exists..")


@router.post("/text2text")
async def llm_textgen(req: Request) -> Dict[str, Any]:
    # dump the received request for debugging purposes    
    logger.info(f"req={req}")

    # initialize vector db and Sagemaker Endpoint
    _init(req)
    
    # now that we have the matching docs, lets pack them as a context
    # into the prompt and ask the LLM to generate a response    
    answer = query_sm_endpoint(req)
    resp = {'question': req.q, 'answer': answer}
    return resp

@router.post("/rag")
async def rag_handler(req: Request) -> Dict[str, Any]:
    # dump the received request for debugging purposes
    logger.info(f"req={req}")

    # initialize vector db and Sagemaker Endpoint
    _init(req)

    # Use the vector db to find similar documents to the query
    # the vector db call would automatically convert the query text
    # into embeddings
    docs = _vector_db.similarity_search(req.q, req.max_matching_docs)
    logger.info(f"here are the {req.max_matching_docs} closest matching docs to the query=\"{req.q}\"")
    for d in docs:
        logger.info(f"---------")
        logger.info(d)
        logger.info(f"---------")

    # now that we have the matching docs, lets pack them as a context
    # into the prompt and ask the LLM to generate a response
    prompt_template = """Answer based on context:\n\n{context}\n\n{question}"""

    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    logger.info(f"prompt sent to llm = \"{prompt}\"")
    chain = load_qa_chain(llm=_sm_llm, prompt=prompt)
    answer = chain({"input_documents": docs, "question": req.q}, return_only_outputs=True)['output_text']
    logger.info(f"answer received from llm,\nquestion: \"{req.q}\"\nanswer: \"{answer}\"")
    resp = {'question': req.q, 'answer': answer}
    if req.verbose is True:
        resp['docs'] = docs
    
    return resp
