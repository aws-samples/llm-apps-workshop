# QA with a fine-tuned LLM and RAG

A question answer task on a corpus of enterprise specific data is a common use-case in an enterprise scenario. If the data to be used for this task is publicly available then chances are that a pre-trained foundation large language model (LLM) will be able to provide a reasonable response to the question but this approach suffers from the following problems: a) the LLM is trained with a point in time snapshot of the data so its response will not be current, b) the LLM could hallucinate i.e. provide convincing looking responses that are factually incorrect and c) most importantly, the model may never have seen the enterprise specific data and is therefore not able to provide a useful response.

All of these problems can be solved by using one of the following approaches:

1. Use Retrieval Augmented Generation (RAG) i.e. consult the enterprise specific knowledge corpus to find specific chunks of data (text) that are likely to contain answers to the question asked and then include this relevant data as context along with the question in the "prompt" provided to the LLM.

1. Fine-tune the LLM on a question answering task using the enterprise specific knowledge corpus and then use RAG. The fine-tuned model now already has better baseline understanding of the enterprise data than the pre-trained LLM and in combination with RAG it can consult the most up to date version of the knowledge corpus to provide the best response to a question.

The following diagram shows a potential architecture of this solution for a virtual agent assist platform.

![](images/finetuning_llm_and_rag.png)

Here is a screenshot of a Chatbot app built on this architecture.
![](images/chatbot.png)

## Installation

1. Create a `conda` environment for `Python 3.9`.

```{{bash}}

conda create -n py39 python=3.9 -y

# activate the environment using `source` or `conda` (whichever one works for your dev platform)
source activate py39
```

1. Package and upload `function.zip` to S3.

```{{bash}}
./deploy.sh
```

1. Update the code for the Lambda function to point to the S3 file uploaded in the step above.

### Testing locally on Windows 10

To build `function.zip` for the Lambda function

```{{bash}}

# from the root directory of the repo
del function.zip
cd .\env\Lib\site-packages\

# download zip.exe from somewhere (I forgot the link)
C:\Users\your-user-name\Downloads\zip -r9 C:\Users\your-user-name\repos\qa-w-rag-finetuned-llm\function.zip .
cd ../../../
C:\Users\your-user-name\Downloads\zip  -g ./function.zip -r app

# upload the function.zip file to s3://qa-w-rag-finetuned-llm/function.zip
aws s3 cp function.zip s3://qa-w-rag-finetuned-llm
```

## Usage

To run the API:
```{{bash}}
uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload
```

To run the chatbot:
```{{bash}}
cd app
streamlit run  webapp.py
```
## Roadmap

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.
