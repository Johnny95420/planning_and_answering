from graph import graph
from states import QuestionStateInput

with open("/react_agent/2303_ver1.md", "r") as f:
    information = f.read()
config = {
    "number_of_queries": 5,
    "use_web": True,
    "use_local_db": False,
    "thread_id": "0",
    "recursion_limit": 50,
}
question = """
現在時間是 2025/05/30，針對聯華電子 2025 年的相關投資報告我有以下問題，請使用2025年並且使用請使用中立角度評斷資訊
請估計聯電 UMC 的 2025 目標價
 * 探查不同風險與有益情況，必須包含 5 ~ 7 個情境 (不同的風險與機會) 下的目標價格
 * 假設不同情境與相關資訊
 * 使用合理並且有理論依據的訂價模型當作基礎
 * 根據定價模型，使用程式,統計,量化風格的方式，模擬與假設估計聯電的 2025 目標價
 * 評估各個情境的可能性，與樂觀程度
"""
async for state in graph.astream(
    QuestionStateInput(question=question, background_knowledge=information),
    config=config,
):
    print(state)
for state in graph.get_state_history(config):
    break
# %%
print(state.values["report"])
# %%
with open("2023_price_detail.md", "w") as f:
    f.write(state.values["report"])
# %%
