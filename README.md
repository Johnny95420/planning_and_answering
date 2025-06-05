# 💰 Planning and Answering

⚠️ This project implements a ReAct-inspired agent system, specialized for in-depth financial, stock, and industry analysis. It features task-tailored prompt engineering for financial contexts and a robust execution flow involving planning, tool use, and iterative refinement.

A modular and automated financial analysis tool designed for comprehensive topic investigation using Large Language Models (LLMs), dynamic planning, tool integration (web search, code execution), and multi-stage report generation. Built with LangGraph for structured, dynamic execution.

This is a follow-up project to Report-Summarizer (https://github.com/Johnny95420/Report-Summarizer). While Report-Summarizer can generate broad reports, it might lack depth when addressing specific questions. Therefore, the purpose of this project is to conduct in-depth research and provide supplementary information for specific sections after Report-Summarizer has finished its execution.

---

## 🚀 Features

* 🧠 **ReAct-Inspired Core:** Employs a "Reasoning and Acting" cycle for problem decomposition and execution.
* 📊 **Iterative Information Processing:** Gathers data via web search, processes it using a `ContentExtractor`, and summarizes conversation history to maintain context.
* 📈 **Multi-Step Planning & Execution:** Breaks down complex financial questions into a sequence of manageable analytical steps, including generating queries, executing steps, and validating the plan.
* 🐍 **Python for Financial Calculations:** Leverages a Python REPL tool (`run_python`) for quantitative analysis, financial modeling (e.g., DDM, DCF, P/E as suggested in `run_python` docstring), and simulations.
* 🔄 **Dynamic Re-planning & Validation:** Assesses gathered information and can dynamically adjust the plan to ensure comprehensive analysis using a `validate_plan` node.
---

## ⚙️ Configuration

**`config.yaml`**:
Defines the language models used by the agent.

```yaml
MODEL_NAME: "deepseek/deepseek-chat"
BACKUP_MODEL_NAME: "gpt-4o-mini"

VERIFY_MODEL_NAME: "deepseek/deepseek-reasoner"
BACKUP_VERIFY_MODEL_NAME: "o4-mini"
```
## 🧪 Usage
The system is designed to be run asynchronously. The main graph compiles with an InMemorySaver for checkpointer, allowing for stateful execution if thread_id is managed.

Configure and Launch the Full Analysis Graph:
The primary way to use the agent for generating a financial report is by invoking the main graph defined in graph.py.

```python
from graph import graph
from states import QuestionStateInput

information = "Some refrence"
config = {
    "number_of_queries": 5,
    "use_web": True,
    "use_local_db": False,
    "thread_id": "0",
    "recursion_limit": 50,
}
question = """ A Question with detail description"""
async for state in graph.astream(
    QuestionStateInput(question=question, background_knowledge=information),
    config=config,
):
    print(state)
with open("report.md", "w") as f:
    f.write(state["report"])
```