# -*- coding: utf-8 -*-
import os
from typing import Optional
import faiss
from langchain import LLMChain, PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.agents import Tool
from langchain.agents import ZeroShotAgent, AgentExecutor
from langchain.utilities import GoogleSearchAPIWrapper
from kube_copilot.shell import KubeProcess
from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore
from langchain.embeddings import OpenAIEmbeddings
from kube_copilot.output_parser import ChatMRKLOutputParser
from kube_copilot.baby_agi import BabyAGI


HUMAN_MESSAGE_TEMPLATE = """Previous steps: {previous_steps}

Current objective: {current_step}

{agent_scratchpad}"""


class CopilotLLM:
    '''Wrapper for LLM chain.'''

    def __init__(self, verbose=True, model="gpt-4", additional_tools=None):
        '''Initialize the LLM chain.'''
        self.chain = get_chat_chain(
            verbose, model, additional_tools=additional_tools)

    def run(self, objective):
        '''Run the LLM chain.'''
        return self.chain({"objective": objective})
        # try:
        #     result = self.chain({"objective": objective})
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
                         max_tokens=4000,
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
        llm = ChatOpenAI(model_name=model, temperature=0, max_tokens=4000)
        embeddings = OpenAIEmbeddings(
            model="text-embedding-ada-002",
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )

    todo_prompt = PromptTemplate.from_template(
        '''You are a planner who is an Kubernetes and cloud native technology
expert at coming up with a todo list for a given objective. Please recheck the
todo list and refine the list into at most 10 items. Come up with a concise todo
list for this objective: {objective}''')
    todo_chain = LLMChain(llm=llm, prompt=todo_prompt)
    llm_chain = LLMChain(llm=llm, prompt=PromptTemplate.from_template(
        '''As a technical expert in Kubernetes and cloud native networking,
your task is to follow the instructions below to complete the required tasks,
ensuring that all actions are within the domains of Kubernetes and cloud native
networking. Ensure that each of your responses is concise and adheres strictly
to the guidelines provided.

Diagnose and provide concise solutions for the question {question}'''))
    tools = [
        Tool(
            name="TODO",
            func=todo_chain.run,
            description="Useful for when you need to come up with todo lists. Input: an objective to create a todo list for. Output: a todo list for that objective. Please be very clear what the objective is!",
        ),
        Tool(
            name="llm",
            func=llm_chain.run,
            description="Useful for when you need to run the LLM chain. Input: question. Output: concise list of issues and solutions.",
        ),
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
                description="Useful for searching the web for current events or current state of the world"
            )
        ]

    if additional_tools is not None:
        tools += additional_tools

    prefix = """You are a Kubernetes and cloud native technology expert who performs one task based on the following objective: {objective}. Take into account these previously completed tasks: {context}."""
    suffix = """Question: {task}
    {agent_scratchpad}"""
    format_instructions = """Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question.

Please note that:
1) the prefix of "Thought: " and "Action: " are must be included.
2) the total number of characters in the input must be refined within 4000 characters.
"""
    prompt = ZeroShotAgent.create_prompt(
        tools,
        prefix=prefix,
        suffix=suffix,
        format_instructions=format_instructions,
        input_variables=["objective", "task", "context", "agent_scratchpad"],
    )

    agent = ZeroShotAgent(llm_chain=LLMChain(llm=llm, prompt=prompt),
                          output_parser=ChatMRKLOutputParser(),
                          allowed_tools=[tool.name for tool in tools])
    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent, tools=tools, verbose=verbose,
    )

    embedding_size = 1536
    index = faiss.IndexFlatL2(embedding_size)
    vectorstore = FAISS(embeddings.embed_query, index,
                        InMemoryDocstore({}), {})
    max_iterations: Optional[int] = 30
    baby_agi = BabyAGI.from_llm(
        llm=llm,
        verbose=verbose,
        vectorstore=vectorstore,
        task_execution_chain=agent_executor,
        max_iterations=max_iterations,
    )

    return baby_agi
