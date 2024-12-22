rm -rf venv
sudo rm -rf litellm
which python3.11
export PATH="/Users/jasper/.pyenv/shims:$PATH"
/Users/jasper/.pyenv/shims/python3.11 -m venv litellm
source litellm/bin/activate
pip install litellm
pip install pydantic
export HUGGINGFACE_API_KEY=your_api_key