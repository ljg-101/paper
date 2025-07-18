import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.prompts import ChatPromptTemplate

# 加载.env文件中的环境变量
load_dotenv()

llm = init_chat_model(
    "deepseek-chat",  # 使用DeepSeek模型
    api_key=os.environ.get("DEEPSEEK_API_KEY")
)


class PolishChain:
    """
    学术润色链。
    """
    def __init__(self):
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_template(
            """
            你是一名专业的学术论文编辑。请对下方句子{content}进行学术化润色，提升语法、逻辑、用词和表达的准确性与流畅性，保持原意不变，不得大幅扩写或缩写。请直接输出润色后的完整句子，不要输出任何解释或多余内容。
            """
        )
        self.chain = self.prompt | self.llm

    def invoke(self, input_data):
        """
        使用提供的输入数据调用链以生成润色结果。
        参数:
            input_data (dict): 包含 'history' 和 'question' 键的字典。
        返回:
            str: 润色后的句子。
        """
        return self.chain.invoke(input_data)