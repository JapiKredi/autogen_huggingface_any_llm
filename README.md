# Remove existing virtual environment
rm -rf venv

# Remove existing litellm directory
sudo rm -rf litellm

# Check the path of Python 3.11
which python3.11

# Update PATH to include pyenv shims
export PATH="/Users/jasper/.pyenv/shims:$PATH"

# Create a new virtual environment named litellm
/Users/jasper/.pyenv/shims/python3.11 -m venv litellm

# Activate the virtual environment
source litellm/bin/activate

# Install litellm package
pip install litellm

# Install pydantic package
pip install pydantic

# Set the Hugging Face API key
export HUGGINGFACE_API_KEY=your_api_key_here

# Echo the Hugging Face API key to verify
echo $HUGGINGFACE_API_KEY

# Set the OpenAI API key
export OPENAI_API_KEY=your_api_key_here

# Echo the OpenAI API key to verify
echo $OPENAI_API_KEY

# Install litellm with proxy support
pip install 'litellm[proxy]'

# Run litellm with the specified model
litellm --model huggingface/Qwen/Qwen2.5-7B-Instruct

