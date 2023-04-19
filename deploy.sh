rm -rf deps
rm -f function.zip
pip install -r app/requirements.txt --no-cache-dir --target=deps
cd deps
rm -rf numpy*
wget https://files.pythonhosted.org/packages/f4/f4/45e6e3f7a23b9023554903a122c95585e9787f9403d386bafb7a95d24c9b/numpy-1.24.2-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl 
unzip numpy-1.24.2-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl  
rm -rf boto*
rm -f numpy-1.24.2-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl  
zip -r9 ../function.zip .
cd -
zip -g ./function.zip -r app
aws s3 cp function.zip s3://qa-w-rag-finetuned-llm
