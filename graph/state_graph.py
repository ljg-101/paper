import os
from dotenv import load_dotenv
from graph.state import State
from chains.check import CheckChain
from chains.polish import PolishChain
from chains.review import ReviewChain
from langchain_core.output_parsers import JsonOutputParser
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage,BaseMessage, HumanMessage
from langchain.prompts import ChatPromptTemplate
from langgraph.graph.state import StateGraph, CompiledStateGraph, END
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

llm = init_chat_model(
    "deepseek-chat",  # 使用DeepSeek模型
    api_key=os.environ.get("DEEPSEEK_API_KEY")
)


def llm_decision(state: State) -> str:
    """
    判断用户输入是否有语法错误，决定流程走向。
    如果有语法错误，进入check节点；否则进入polish节点。
    """
    user_message = state["messages"][-1].content  # 获取最新一条用户输入

    # 组织 prompt，询问模型是否有语法错误
    prompt = (
        "你是一个语法助手。请判断用户输入是否存在语法错误。\n"
        f"用户输入：{user_message}\n"
        "如果有语法错误，回答 'check'；如果没有语法问题，回答 'polish'。\n"
        "请只返回一个单词：check 或 polish。"
    )

    # 调用 LLM 获取回复
    response = llm.invoke(prompt)
    answer = response.content.strip().lower()  # strip()去字符.lower()变小写

    if answer not in ["check", "polish"]:
        raise ValueError(f"模型输出不合法: {answer}")

    return answer


def check(state: State) -> State:
    """
    check节点：对输入内容进行语法、拼写、标点纠错。
    """
    print("--- CHECKING ---")
    chain = CheckChain()
    messages = state["messages"]
    state["messages"] = chain.invoke({"content": messages[-1].content, "history": messages[:-1]})
    return state


def polish(state: State) -> State:
    """
    polish节点：对输入内容进行学术化润色，提升表达质量。
    """
    print("--- POLISHINGING ---")
    chain = PolishChain()
    messages = state["messages"]
    state["messages"] = chain.invoke({"content": messages[-1].content, "history": messages[:-1]})
    return state


def review(state: State) -> State:
    """
    review节点：模拟SCI审稿专家，对内容进行总结和提出专业建议。
    """
    print("--- REVIEWING ---")
    chain = ReviewChain()
    messages = state["messages"]
    state["messages"] = chain.invoke({"content": messages[-1].content, "history": messages[:-1]})
    return state


def create_graph() -> CompiledStateGraph:
    """
    创建并配置状态图工作流。

    返回:
        CompiledStateGraph: 编译好的状态图
    """

    workflow = StateGraph(State)
    # 添加节点
    workflow.add_node("check", check)
    workflow.add_node("polish", polish)
    workflow.add_node("review", review)
    # 添加边
    workflow.set_conditional_entry_point(
        llm_decision,
        {
            "check": "check",
            "polish": "polish",
        },
    )
    workflow.add_edge("check", "polish")
    workflow.add_edge("polish", "review")
    workflow.add_edge("review", END)

    # 创建图，并使用 `MemorySaver()` 在内存中保存状态
    return workflow.compile(checkpointer=MemorySaver())


def process_text_units(units, graph, config):
    """
    批量处理文本单元，只收集polish和review节点的AI回复内容。
    :param units: 文本单元列表
    :param graph: 已编译的状态图
    :param config: 配置
    :return: 每个单元的polish和review结果列表（字典）
    """
    results = []
    total = len(units)
    for idx, unit in enumerate(units, 1):
        print(f"\n正在处理第{idx}/{total}段落...")
        state = {"messages": [HumanMessage(content=unit)]}
        polish_content = None
        review_content = None
        for event in graph.stream(state, config):
            for key, value in event.items():
                msg = value.get("messages")
                if key == "polish" and hasattr(msg, "content"):
                    polish_content = msg.content
                if key == "review" and hasattr(msg, "content"):
                    review_content = msg.content
        results.append({
            "polish": polish_content if polish_content else "",
            "review": review_content if review_content else ""
        })
        print(f"已完成{idx}/{total}段落")
    return results


def stream_graph_updates(graph: CompiledStateGraph, user_input: str, config: dict):
    """
    用于终端交互模式：实时流式输出AI和用户的对话内容。
    """
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}, config):
        for value in event.values():
            msg = value.get("messages")
            if hasattr(msg, "content") and msg.__class__.__name__ == "AIMessage":
                print(f"AI: {msg.content}")
            elif hasattr(msg, "content") and msg.__class__.__name__ == "HumanMessage":
                print(f"User: {msg.content}")

