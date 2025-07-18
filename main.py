import uuid
from dotenv import load_dotenv
from langchain.schema import Document
from langchain_core.messages import AIMessage, HumanMessage
from graph.state_graph import create_graph, stream_graph_updates, State, process_text_units
from utils.text_splitter import split_text
from utils.file_utils import read_text_file


def main():
    load_dotenv(verbose=True)
    config = {"configurable": {"thread_id": uuid.uuid4().hex}}
    graph = create_graph()

    print("请选择输入方式：1-终端输入 2-文件输入")
    mode = input("输入1或2: ").strip()
    if mode == '2':
        file_path = input("请输入文件路径: ").strip()
        text = read_text_file(file_path)
        units = split_text(text, mode='section')
        results = process_text_units(units, graph, config)
        with open('result.txt', 'w', encoding='utf-8') as f:
            for idx, res in enumerate(results, 1):
                f.write(f"【第{idx}段优化结果】\n{res}\n\n")
                polish_text = res['polish'].strip() if res['polish'] else ''
                review_text = res['review'].strip() if res['review'] else ''
                f.write(f"{polish_text}\n{review_text}\n\n")
        print("\n全部处理完成，结果已保存到 result.txt")
    else:
        while True:
            user_input = input("User: ")
            if user_input.lower() in ["exit", "quit", "q"]:
                break
            stream_graph_updates(graph, user_input, config)

if __name__ == "__main__":
    main()
