# QA with LLM and RAG

A question answer task on a corpus of enterprise specific data is a common use-case in an enterprise scenario. If the data to be used for this task is publicly available then chances are that a pre-trained foundation large language model (LLM) will be able to provide a reasonable response to the question but this approach suffers from the following problems: a) the LLM is trained with a point in time snapshot of the data so its response will not be current, b) the LLM could hallucinate i.e. provide convincing looking responses that are factually incorrect and c) most importantly, the model may never have seen the enterprise specific data and is therefore not able to provide a useful response.

All of these problems can be solved by using one of the following approaches:

1. Use Retrieval Augmented Generation (RAG) i.e. consult the enterprise specific knowledge corpus to find specific chunks of data (text) that are likely to contain answers to the question asked and then include this relevant data as context along with the question in the "prompt" provided to the LLM.

1. As an additional step we could also Fine-tune the LLM on a question answering task using the enterprise specific knowledge corpus and then use RAG. The fine-tuned model now already has better baseline understanding of the enterprise data than the pre-trained LLM and in combination with RAG it can consult the most up to date version of the knowledge corpus to provide the best response to a question.

The following diagram shows a potential architecture of this solution for a virtual agent assist platform.

![](images/finetuning_llm_and_rag.png)

Here is a screenshot of a Chatbot app built on this architecture.
![](images/chatbot.png)

## Installation

Follow the steps listed below to create and run the RAG solution. The [blog_post.md](./blog_post.md) describes this solution in detail.

1. Launch the cloud formation template included in this repository using one of the buttons from the table below. The cloud formation template will create an IAM role called `LLMAppsBlogIAMRole` and a SageMaker Notebook called `aws-llm-apps-blog` that we will use for running the code in this repository. This cloud formation template creates the Amazon OpenSearch cluster, LLM endpoints for text generation and embeddings and a Amazon SageMaker Notebook with this repository cloned to run the next steps.


   |AWS Region                |     Link        |
   |:------------------------:|:-----------:|
   |us-east-1 (N. Virginia)    | [<img src="./img/ML-14328-cloudformation-launch-stack.png">](https://console.aws.amazon.com/cloudformation/home?region=us-east-1#/stacks/new?stackName=llm-apps-blog-rag&templateURL=https://aws-blogs-artifacts-public.s3.amazonaws.com/artifacts/ML-14328/template.yml) |
   |us-west-2 (Oregon)          | [<img src="./img/ML-14328-cloudformation-launch-stack.png">](https://console.aws.amazon.com/cloudformation/home?region=us-west-2#/stacks/new?stackName=llm-apps-blog-rag&templateURL=https://aws-blogs-artifacts-public.s3.amazonaws.com/artifacts/ML-14328/template.yml) |
   |eu-west-1 (Dublin)        | [<img src="./img/ML-14328-cloudformation-launch-stack.png">](https://console.aws.amazon.com/cloudformation/home?region=eu-west-1#/stacks/new?stackName=llm-apps-blog-rag&templateURL=https://aws-blogs-artifacts-public.s3.amazonaws.com/artifacts/ML-14328/template.yml) |
   |ap-northeast-1 (Tokyo)    | [<img src="./img/ML-14328-cloudformation-launch-stack.png">](https://console.aws.amazon.com/cloudformation/home?region=ap-northeast-1#/stacks/new?stackName=llm-apps-blog-rag&templateURL=https://aws-blogs-artifacts-public.s3.amazonaws.com/artifacts/ML-14328/template.yml) |

1. Once the cloud formation stack has been created successfully, open the Outputs tab of the stack and note the URL for the API Gateway we will be needing it to the run a RAG query later on.

1. Open the `aws-llm-apps-blog` SageMaker Notebook created by the cloud formation template and then find the [`data_ingestion_to_vectordb.ipynb`](data_ingestion_to_vectordb.ipynb) file and double click on it.

1. Do a `Run All` for this notebook. It will ingest the data (embeddings) into the OpenSearch cluster and once that is done, we are now ready to ask some questions via the `/rag` endpoint of the Lambda function.

1. Query the API Gateway `/rag` endpoint using the following command. The endpoint can be seen on the Outputs tab of the cloud formation stack, it is value of the `LLMAppAPIEndpoint` key.

    ```{{bash}}
    curl -X POST "https://replace-with-your-api-gw-url/prod/api/v1/llm/rag" -H  "accept: application/json" -H  "Content-Type: application/json" -d "{\"q\":\"Which versions of XGBoost does SageMaker support?\"}"
    ```
1. Run the [`streamlit`](https://streamlit.io/) app for the chatbot on `SageMaker Studio`. On `Sagemaker Studio` create a new `Terminal` and run the following commands:

    ```{{bash}}
    git clone https://github.com/aws-samples/llm-apps-workshop    
    cd llm-apps-workshop/blogs/rag/app
    pip install -r requirements.txt
    streamlit run webapp.py    
    ```
    This will start a streamlit app on SageMaker Studio, you can access the app by opening the following URL in a new browser tab `https://replace-with-your-studio-domain.studio.replace-with-your-region.sagemaker.aws/jupyter/default/proxy/8501/webapp`

### Building your version of the Lambda

1. Open a new Terminal on the SageMaker Notebook and change to the `rag/api` directory using the following command:

    ```
    cd /home/ec2-user/SageMaker/repos/llm-apps-workshop/blogs/rag/api
    ```

1. Create a `conda` environment for `Python 3.9`.

    ```{{bash}}

    conda create -n py39 python=3.9 -y

    # activate the environment
    source activate py39
    ```

1. Package and upload `function.zip` to the SageMaker bucket for your region.

    ```{{bash}}
    ./deploy.sh
    ```

1. Update the code for the Lambda function to point to the S3 file uploaded in the step above.
