# -*- coding: utf-8 -*-
import os
import faiss
from langchain.chat_models import ChatOpenAI
from langchain.agents import Tool
from langchain.utilities import GoogleSearchAPIWrapper
from kube_copilot.shell import KubeProcess
from langchain.experimental import AutoGPT
from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore
from langchain.embeddings import OpenAIEmbeddings


HUMAN_MESSAGE_TEMPLATE = """Previous steps: {previous_steps}

Current objective: {current_step}

{agent_scratchpad}"""


class CopilotLLM:
    '''Wrapper for LLM chain.'''

    def __init__(self, verbose=True, model="gpt-4", additional_tools=None):
        '''Initialize the LLM chain.'''
        self.chain = get_chat_chain(
            verbose, model, additional_tools=additional_tools)

    def run(self, instructions):
        '''Run the LLM chain.'''
        return self.chain.run([instructions])
        # try:
        #     result = self.chain.run(instructions)
        #     return result
        # except Exception as e:
        #     # TODO: Workaround for issue https://github.com/hwchase17/langchain/issues/1358.
        #     if "Could not parse LLM output:" in str(e):
        #         return str(e).removeprefix("Could not parse LLM output: `").removesuffix("`")
        #     else:
        #         raise e


def get_chat_chain(verbose=True, model="gpt-3.5-turbo", additional_tools=None):
    '''Initialize the LLM chain with useful tools.'''
    if os.getenv("OPENAI_API_TYPE") == "azure":
        engine = model.replace(".", "")
        llm = ChatOpenAI(model_name=model, temperature=0,
                         model_kwargs={"engine": engine})
        embeddings = OpenAIEmbeddings(
            deployment="text-embedding-ada-002",
            model="text-embedding-ada-002",
            openai_api_type=os.getenv("OPENAI_API_TYPE"),
            openai_api_base=os.getenv("OPENAI_API_BASE"),
            openai_api_version="2023-05-15",
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            chunk_size=1  # The max number of inputs is 1 for Azure OpenAI
        )
    else:
        llm = ChatOpenAI(model_name=model, temperature=0)
        embeddings = OpenAIEmbeddings(
            deployment="text-embedding-ada-002",
            model="text-embedding-ada-002",
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )

    tools = [
        Tool(
            name="kubectl",
            description="Useful for executing kubectl command to query information from kubernetes cluster. Input: a kubectl get command. Output: the yaml for the resource.",
            func=KubeProcess(command="kubectl").run,
        ),
        Tool(
            name="trivy",
            description="Useful for executing trivy image command to scan images for vulnerabilities. Input: a trivy image command. Output: the vulnerabilities found in the image.",
            func=KubeProcess(command="trivy").run,
        ),
    ]

    if os.getenv("GOOGLE_API_KEY") and os.getenv("GOOGLE_CSE_ID"):
        tools += [
            Tool(
                name="search",
                func=GoogleSearchAPIWrapper(
                    google_api_key=os.getenv("GOOGLE_API_KEY"),
                    google_cse_id=os.getenv("GOOGLE_CSE_ID"),
                ).run,
                description="search the web for current events or current state of the world"
            )
        ]

    if additional_tools is not None:
        tools += additional_tools

    embedding_size = 1536
    index = faiss.IndexFlatL2(embedding_size)
    vectorstore = FAISS(embeddings.embed_query, index,
                        InMemoryDocstore({}), {})

    autogpt = AutoGPT.from_llm_and_tools(
        ai_name="kube-copilot",
        ai_role="Assistant",
        tools=tools,
        llm=llm,
        memory=vectorstore.as_retriever(),
        # handle_parsing_errors="Check your output and make sure it conforms!",
    )
    autogpt.chain.verbose = verbose
    return autogpt
