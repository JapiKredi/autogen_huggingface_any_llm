import autogen
from autogen import (
    AssistantAgent,
    UserProxyAgent,
    GroupChatManager,
    GroupChat,
    ConversableAgent,
)
from transformers import AutoTokenizer, GenerationConfig, AutoModelForCausalLM
from types import SimpleNamespace
import requests
import json
import os
import shutil
import random


class APIModelClient:
    def __init__(self, config, **kwargs):
        self.device = config.get("device", "cpu")
        self.api_url = "https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1"
        self.headers = {"Authorization": "Bearer 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"}

        self.model_name = config.get("model")
        self.chat_index = 0
        self.conversion_mem = ""

    def create(self, params):
        conversation_history = ""

        for message in params["messages"]:
            prefix = ""
            if message["role"] == "system":
                prefix = f"Bot Description:\n"
            elif message["role"] == "user":
                prefix = f"User____:\n"
            else:
                prefix = f'Agent ({message["role"]}):\n'
            conversation_history += prefix + f'{message["content"]}\n\n'

        input_data = {
            "inputs": conversation_history,
            "parameters": {
                "max_new_tokens": 1000,
                "return_full_text": False,
                "do_sample": False,
            },
            "options": {"wait_for_model": True, "use_cache": False},
        }

        response = requests.post(self.api_url, json=input_data, headers=self.headers)

        if response.status_code == 200:
            api_response = response.json()
            if "\n    ```" in api_response[0]["generated_text"]:
                api_response[0]["generated_text"] = api_response[0][
                    "generated_text"
                ].replace("\n    ", "\n")
            model_response = SimpleNamespace()
            model_response.choices = []

            choice = SimpleNamespace()
            choice.message = SimpleNamespace(content=api_response[0]["generated_text"])
            model_response.choices.append(choice)
            model_response.model = self.model_name
            self.chat_index += 1
            return model_response
        else:
            raise Exception(
                f"API request failed with status code {response.status_code}: {response.text}"
            )

    def message_retrieval(self, response):
        choices = response.choices
        return [choice.message.content for choice in choices]

    def cost(self, response) -> float:
        response.cost = 0
        return 0

    @staticmethod
    def get_usage(response):
        return {}


class APIModelClientWithArguments(APIModelClient):
    def __init__(self, config, hf_key, hf_url, **kwargs):
        super().__init__(config, **kwargs)
        self.api_url = hf_url
        self.headers = {"Authorization": f"Bearer {hf_key}"}


def UserAgent(
    name,
    hf_key,
    hf_url,
    max_consecutive_auto_reply=2,
    code_dir="coding",
    use_docker=False,
    system_message="You are a helpful AI assistant",
):
    llm_config = {
        "config_list": [
            {
                "model": "",
                "model_client_cls": "APIModelClientWithArguments",
                "device": "",
            }
        ]
    }
    user_agent = UserProxyAgent(
        name="user_proxy",
        max_consecutive_auto_reply=max_consecutive_auto_reply,
        llm_config=llm_config,
        is_termination_msg=lambda x: x.get("content", "")
        .rstrip()
        .endswith("TERMINATE"),
        code_execution_config={
            "work_dir": code_dir,
            "use_docker": use_docker,
        },
        system_message=system_message,
        human_input_mode="TERMINATE",
    )
    user_agent.register_model_client(
        model_client_cls=APIModelClientWithArguments, hf_key=hf_key, hf_url=hf_url
    )
    return user_agent


def ModelAgent(
    name,
    hf_key,
    hf_url="https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1",
    system_message="",
    code_execution=False,
):
    default_system_message = """You are a helpful AI assistant.
    Solve tasks using your coding and language skills.
    In the following cases, suggest python code (in a python coding block) or shell script (in a sh coding block) for the user to execute. Make sure to prefix the code block with 'python' or 'sh' depending.
        1. When you need to collect info, use the code to output the info you need, for example, browse or search the web, download/read a file, print the content of a webpage or a file, get the current date/time, check the operating system. After sufficient info is printed and the task is ready to be solved based on your language skill, you can solve the task by yourself.
        2. When you need to perform some task with code, use the code to perform the task and output the result. Finish the task smartly.
    Solve the task step by step if you need to. If a plan is not provided, explain your plan first. Be clear which step uses code, and which step uses your language skill.
    When using code, you must indicate the script type in the code block. The user cannot provide any other feedback or perform any other action beyond executing the code you suggest. The user can't modify your code. So do not suggest incomplete code which requires users to modify. Don't use a code block if it's not intended to be executed by the user.
    If you want the user to save the code in a file before executing it, put # filename: <filename> inside the code block as the first line. Don't include multiple code blocks in one response. Do not ask users to copy and paste the result. Instead, use 'print' function for the output when relevant. Check the execution result returned by the user.
    If the result indicates there is an error, fix the error and output the code again. Suggest the full code instead of partial code or code changes. If the error can't be fixed or if the task is not solved even after the code is executed successfully, analyze the problem, revisit your assumption, collect additional info you need, and think of a different approach to try..
    When you find an answer, verify the answer carefully. Include verifiable evidence in your response if possible.
    """

    if system_message == "":
        system_message = default_system_message

    llm_config = {
        "config_list": [
            {
                "model": "",
                "model_client_cls": "APIModelClientWithArguments",
                "device": "",
            }
        ]
    }

    agent = ConversableAgent(
        name=name,
        llm_config=llm_config,
        system_message=system_message,
        code_execution_config=code_execution,
    )
    agent.register_model_client(
        model_client_cls=APIModelClientWithArguments, hf_key=hf_key, hf_url=hf_url
    )

    return agent


def InitChat(user, agent, _input, summary_method="reflection_with_llm"):
    def clear_directory_contents(dir_path):
        try:
            for item in os.listdir(dir_path):
                item_path = os.path.join(dir_path, item)
                if os.path.isfile(item_path) or os.path.islink(item_path):
                    os.remove(item_path)  # Remove files and links
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)  # Remove directories
            shutil.rmtree(dir_path)
            print(f"All contents of '{dir_path}' have been removed.")
        except FileNotFoundError:
            pass

    seed = 42
    clear_directory_contents(f"./autogen_cache/{seed}")

    custom_cache = autogen.Cache(
        {"cache_seed": seed, "cache_path_root": "autogen_cache"}
    )

    chat_res = user.initiate_chat(
        agent,
        message=_input,
        summary_method=summary_method,
        cache=custom_cache,
    )

    clear_directory_contents(f"./autogen_cache/{seed}")


def GroupChat(
    user,
    agents,
    _input,
    hf_key,
    hf_url="https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1",
    max_round=5,
):
    llm_config = {
        "config_list": [
            {
                "model": "",
                "model_client_cls": "APIModelClientWithArguments",
                "device": "",
            }
        ]
    }

    groupchat = autogen.GroupChat(
        agents=agents,
        messages=[],
        max_round=max_round,
        speaker_selection_method="round_robin",
        allow_repeat_speaker=False,
    )
    manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

    manager.register_model_client(
        model_client_cls=APIModelClientWithArguments, hf_key=hf_key, hf_url=hf_url
    )
    InitChat(user, manager, _input)


if __name__ == "__main__":
    _input = input("How can I help my sons to better prepare for their exams???.\n")
    hf_key = "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"
    hf_url = "https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1"
    user = UserAgent("user_proxy", hf_key=hf_key, hf_url=hf_url)
    assistant = ModelAgent(
        "assistant", hf_key, hf_url, system_message="You are a friendly AI assistant."
    )
    InitChat(user, assistant, _input)
