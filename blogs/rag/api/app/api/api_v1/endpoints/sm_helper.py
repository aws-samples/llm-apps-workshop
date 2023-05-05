
import json
import boto3
import logging
from typing import List, Dict
from .fastapi_request import (Request,
                              SAGEMAKER_ENDPOINT_MAPPING)

logger = logging.getLogger(__name__)

def query_endpoint_with_json_payload(encoded_json, endpoint_name, content_type="application/json") -> Dict:
    client = boto3.client("runtime.sagemaker")
    response = client.invoke_endpoint(
        EndpointName=endpoint_name, ContentType=content_type, Body=encoded_json
    )
    return response

def parse_response_model_flan_t5(query_response) -> List:
    model_predictions = json.loads(query_response["Body"].read())
    generated_text = model_predictions["generated_texts"]
    return generated_text

def query_sm_endpoint(req: Request) -> List:
    payload = {
        "text_inputs": req.q,
        "max_length": req.max_length,
        "num_return_sequences": req.num_return_sequences,
        "top_k": req.top_k,
        "top_p": req.top_p,
        "do_sample": req.do_sample,
    }

    endpoint_name = req.text_generation_model
    query_response = query_endpoint_with_json_payload(
            json.dumps(payload).encode("utf-8"), endpoint_name=SAGEMAKER_ENDPOINT_MAPPING[endpoint_name]
        )

    generated_texts = parse_response_model_flan_t5(query_response)
    logger.info(f"the generated output is: {generated_texts}")
    return generated_texts
