python -m venv venv
source venv/bin/activate
pip install litellm
pip install pydantic
pip install litellm
export HUGGINGFACE_API_KEY=your_api_key_here
echo $HUGGINGFACE_API_KEY
pip install 'litellm[proxy]'
litellm --model huggingface/Qwen/Qwen2.5-7B-Instruct