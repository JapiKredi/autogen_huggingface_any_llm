rm -rf venv
sudo rm -rf litellm
which python3.11
export PATH="/Users/jasper/.pyenv/shims:$PATH"
/Users/jasper/.pyenv/shims/python3.11 -m venv litellm
source litellm/bin/activate
pip install litellm
pip install pydantic
export HUGGINGFACE_API_KEY=your_api_key_here
echo $HUGGINGFACE_API_KEY
export OPENAI_API_KEY=your_api_key_here
echo $OPENAI_API_KEY
pip install 'litellm[proxy]'
litellm --model huggingface/Qwen/Qwen2.5-7B-Instruct