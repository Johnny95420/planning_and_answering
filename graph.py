# %%
import asyncio
import logging
from collections import deque
from typing import Dict, List, Literal

from dotenv import load_dotenv
from langchain_community.chat_models import ChatLiteLLM
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Command
from omegaconf import OmegaConf

from prompt import (
    information_query_instruction,
    planner_query_writer_instructions,
    react_compress_prompt,
    react_prompt,
    replanner_instruction,
    report_writer_instructions,
    step_planner_instruction,
    step_writer_instructions,
)
from states import InputState, QuestionState, QuestionStateInput, State, Step
from tools import (
    queries_formatter,
    run_python,
    selenium_api_search,
    step_formatter,
    validate_formatter,
)

config = OmegaConf.load("config.yaml")

MODEL_NAME = config["MODEL_NAME"]
BACKUP_MODEL_NAME = config["BACKUP_MODEL_NAME"]

VERIFY_MODEL_NAME = config["VERIFY_MODEL_NAME"]
BACKUP_VERIFY_MODEL_NAME = config["BACKUP_VERIFY_MODEL_NAME"]
load_dotenv(".env")
# %%
logger = logging.getLogger("AgentLogger")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)


formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


# %%
def call_llm(prompt: List, tool=None, tool_choice=None):
    try:
        model = ChatLiteLLM(model=MODEL_NAME, temperature=0)
        if tool:
            model = model.bind_tools(tools=tool, tool_choice=tool_choice)
        response = model.invoke(prompt)
    except Exception as e:
        logger.error(e)
        model = ChatLiteLLM(model=BACKUP_MODEL_NAME, temperature=0)
        if tool:
            model = model.bind_tools(tools=tool, tool_choice=tool_choice)
        response = model.invoke(prompt)
    return response


def call_thinking_llm(prompt: List):
    temperature = (
        1 if VERIFY_MODEL_NAME.startswith("o3") or VERIFY_MODEL_NAME.startswith("o4") else 0
    )
    try:
        model = ChatLiteLLM(model=VERIFY_MODEL_NAME, temperature=temperature)
        response = model.invoke(prompt)

    except Exception as e:
        logger.error(e)
        temperature = (
            1
            if BACKUP_VERIFY_MODEL_NAME.startswith("o3")
            or BACKUP_VERIFY_MODEL_NAME.startswith("o4")
            else 0
        )
        model = ChatLiteLLM(model=BACKUP_VERIFY_MODEL_NAME, temperature=temperature)
        response = model.invoke(prompt)
    return response


TOOLS = [selenium_api_search, run_python]


def call_model(state: State, config: RunnableConfig) -> Dict[str, List[AIMessage]]:
    configurable = config["configurable"]
    n_iters = configurable["n_iters"]
    prev_idx = state.prev_summary_index
    topic = state.topic
    history = state.history
    iterations = state.iterations
    if prev_idx is None:
        prev_idx = 0
        history = history
    else:
        conversation = ""
        for message in state.messages[prev_idx:]:
            conversation += str(message)
        prev_idx = len(state.messages)

        compress_prompt = react_compress_prompt.format(topic=topic, conversation=conversation)
        response = call_llm(
            prompt=[
                SystemMessage(content=compress_prompt),
                HumanMessage(
                    content="Please help me to compress this conversation and preserve the detail about the topic related detail information."
                ),
            ]
        )
        history += "===============\n" + response.content

    if iterations < n_iters:
        prompt = react_prompt.format(topic=topic, history=history)
        response = call_llm(
            prompt=[
                SystemMessage(content=prompt),
                HumanMessage(
                    content="""Please help me to answer ,generate a report or use other tools to do computing about this topic or topic related information.
                    If this question or key information can solve or get by code use tool.
                    Please select the suitable tool(run_python) not only perform query or just output text."""
                ),
            ],
            tool=TOOLS,
            tool_choice="auto",
        )
    else:
        prompt = react_prompt.format(topic=topic, history=history)
        response = call_llm(
            [
                SystemMessage(content=prompt),
                HumanMessage(
                    content="Please according the history information give me a final answer / report."
                ),
            ]
        )
    return {
        "messages": [response],
        "prev_summary_index": prev_idx,
        "history": history,
        "iterations": iterations + 1,
    }


def route_model_output(state: State, config: RunnableConfig) -> Literal["__end__", "tools"]:
    last_message = state.messages[-1]
    if not isinstance(last_message, AIMessage):
        raise ValueError(
            f"Expected AIMessage in output edges, but got {type(last_message).__name__}"
        )
    # If there is no tool call, then we finish
    if not last_message.tool_calls:
        return "__end__"
    # Otherwise we execute the requested actions
    return "tools"


react_agent_builder = StateGraph(State, input=InputState)
react_agent_builder.add_node(call_model)
react_agent_builder.add_node("tools", ToolNode(TOOLS))

react_agent_builder.add_edge("__start__", "call_model")
react_agent_builder.add_conditional_edges("call_model", route_model_output)
react_agent_builder.add_edge("tools", "call_model")
react_agent_graph = react_agent_builder.compile(name="ReAct Agent")


async def question_handler(questions, history="", n_iters: int = 5):
    tasks = [
        react_agent_graph.ainvoke(
            InputState(
                topic=q,
                prev_summary_index=None,
                history=history,
                messages=[HumanMessage(content=q)],
                iterations=0,
            ),
            config={"n_iters": n_iters},
        )
        for q in questions
    ]
    results = await asyncio.gather(*tasks)
    all_answers = [
        {"question": q, "answer": result["messages"][-1].content}
        for q, result in zip(questions, results)
    ]
    return {"qa_pairs": all_answers}


# %%
def str_agg(qa_pairs):
    source_str = ""
    for pair in qa_pairs:
        source_str += "==== Question ====\n"
        source_str += pair["question"] + "\n"
        source_str += "==== Answer ====\n"
        source_str += pair["answer"] + "\n"
    return source_str


async def generate_answering_plan(state: QuestionState, config: RunnableConfig):
    question = state["question"]
    information = state["background_knowledge"]

    configurable = config["configurable"]
    number_of_queries = configurable["number_of_queries"]
    system_instructions_query = planner_query_writer_instructions.format(
        question=question,
        number_of_queries=number_of_queries,
    )
    results = call_llm(
        prompt=[SystemMessage(content=system_instructions_query)]
        + [
            HumanMessage(
                content="Generate queries that will help with planning the anwersing steps."
            )
        ],
        tool=[queries_formatter],
        tool_choice="required",
    )
    answer = await question_handler(
        results.tool_calls[0]["args"]["queries"], history=information, n_iters=3
    )
    source_str = str_agg(answer["qa_pairs"])
    step_planner_inst = step_planner_instruction.format(
        step_min=5,
        step_max=7,
        question=question,
        report_organization="",
        context=source_str,
    )
    response = call_thinking_llm(
        [SystemMessage(content=step_planner_inst)]
        + [
            HumanMessage(
                content="Generate a series of steps that will help with solving the question."
            )
        ],
    )
    output = call_llm(
        [
            SystemMessage(
                content="please help me to convert these planning steps into suitable function call format."
            )
        ]
        + [HumanMessage(content=response.content)],
        tool=[step_formatter],
        tool_choice="required",
    )
    steps = deque([])
    for step in output.tool_calls[0]["args"]["steps"]:
        steps.append(Step(goal=step, content=""))

    return {
        "steps": steps,
        "background_knowledge": information + source_str,
        "completed_steps": [],
        "can_answer": False,
        "plan_solving_iterations": 0,
        "report": "",
    }


async def write_report(state: QuestionState, config: RunnableConfig):
    question = state["question"]
    information = state["background_knowledge"]
    report = state["report"]

    report_writer = report_writer_instructions.format(
        topic=question, section_content=report, context=information
    )
    reponse = call_llm(
        prompt=[SystemMessage(content=report_writer)]
        + [
            HumanMessage(
                content="Based on the content of past reports and additional information, please help me write a detailed, information-complete, comprehensive, and topic-relevant report."
            )
        ]
    )
    report = reponse.content
    information = report
    return {
        "report": report,
    }


async def execute_plan(state: QuestionState, config: RunnableConfig):
    question = state["question"]
    step = state["steps"].popleft()
    current_step = step.goal
    information = state["background_knowledge"]
    configurable = config["configurable"]
    number_of_queries = configurable["number_of_queries"]
    report = state["report"]

    report_writer = report_writer_instructions.format(
        topic=question, section_content=report, context=information
    )
    reponse = call_llm(
        [SystemMessage(content=report_writer)]
        + [
            HumanMessage(
                content="Based on the content of past reports and additional information, please help me write a detailed, information-complete, comprehensive, and topic-relevant report."
            )
        ]
    )
    report = reponse.content

    # * aggregate some inforamtion for this step
    information_query = information_query_instruction.format(
        question=question,
        step=current_step,
        number_of_queries=number_of_queries,
        context=information,
    )
    results = call_llm(
        [SystemMessage(content=information_query)]
        + [HumanMessage(content="Generate queries that will help with solving current step.")],
        tool=[queries_formatter],
        tool_choice="required",
    )
    agg_infos = await question_handler(
        results.tool_calls[0]["args"]["queries"],
        history=information + report,
        n_iters=3,
    )
    info_str = str_agg(agg_infos["qa_pairs"])

    answer = await question_handler(
        [current_step],
        history=information + "\n" + report + "\n" + info_str,
        n_iters=10,
    )
    source_str = str_agg(answer["qa_pairs"])
    step.content = source_str
    information = report + "\n" + info_str + "\n" + source_str

    step_writer = step_writer_instructions.format(
        question=question, step=current_step, context=information
    )

    response = call_llm(
        [SystemMessage(content=step_writer)]
        + [
            HumanMessage(
                content="Please aggregate information for solving this step and answer my question with a detail report."
            )
        ]
    )
    information += (
        "==== Question ====\n"
        + current_step
        + "\n"
        + "==== Answer ====\n"
        + response.content
        + "\n"
    )

    return {
        "steps": state["steps"],
        "background_knowledge": information,
        "completed_steps": [step],
        "report": report,
    }


def validate_plan(state: QuestionState, config: RunnableConfig):
    question = state["question"]
    curr_steps = state["steps"]
    information = state["background_knowledge"]
    num_iterations = state["plan_solving_iterations"]
    step_list = ""
    for step in curr_steps:
        step_list += step.goal
        step_list += "\n"

    replanner = replanner_instruction.format(
        step_min=0,
        step_max=5,
        question=question,
        context=information,
        steps=step_list,
    )
    response = call_thinking_llm(
        [SystemMessage(content=replanner)]
        + [
            HumanMessage(
                content="""Generate a series of steps that will help with solving the question."""
            )
        ],
    )
    response = call_llm(
        prompt=[
            SystemMessage(
                content="please help me to convert these planning steps into suitable function call format."
            )
        ]
        + [HumanMessage(content=response.content)],
        tool=[validate_formatter],
        tool_choice="required",
    )

    if not response.tool_calls[0]["args"]["done"] and num_iterations < 15:
        new_steps = deque([])
        for new_step in response.tool_calls[0]["args"]["new_plan"]:
            new_steps.append(Step(goal=new_step, content=""))

        check = True
        while check:
            count = 0
            for prev_step in state["completed_steps"]:
                if new_steps[0].goal == prev_step.goal:
                    count += 1
            if count >= 2:
                new_steps.popleft()
            else:
                check = False

        return Command(
            goto="execute_plan",
            update={
                "steps": new_steps,
                "plan_solving_iterations": num_iterations + 1,
            },
        )

    else:
        return Command(goto="write_report", update={"plan_solving_iterations": num_iterations + 1})


# %%
builder = StateGraph(QuestionState, input=QuestionStateInput)
builder.add_node("generate_answering_plan", generate_answering_plan)
builder.add_node("execute_plan", execute_plan)
builder.add_node("validate_plan", validate_plan)
builder.add_node("write_report", write_report)

builder.add_edge(START, "generate_answering_plan")
builder.add_edge("generate_answering_plan", "execute_plan")
builder.add_edge("execute_plan", "validate_plan")
builder.add_edge("write_report", END)
checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)
# %%
