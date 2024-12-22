from hfautogen import ModelAgent, UserAgent, InitChat

_input = input("How can we connect AutoGen with models from HuggingFace?.\n")
hf_key = "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"

user = UserAgent("user_proxy", hf_key=hf_key)
assistant = ModelAgent(
    "assistant", hf_key, system_message="You are a friendly AI assistant."
)

InitChat(user, assistant, _input)
