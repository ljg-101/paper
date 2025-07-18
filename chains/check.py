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


class CheckChain:
    """
    学术语法纠错链。
    """
    def __init__(self):
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_template(
            """
            你是一名专业的学术论文编辑。请仔细检查下方句子{content}，仅修正其中的语法、拼写或标点错误，保持原意不变，不得添加、删减或改写内容。请直接输出修改后的完整句子，不要输出任何解释或多余内容。
            """
        )
        self.chain = self.prompt | self.llm

    def invoke(self, input_data):
        """
        使用提供的输入数据调用链以生成纠错结果。
        参数:
            input_data (dict): 包含 'history' 和 'question' 键的字典。
        返回:
            str: 纠错后的句子。
        """
        return self.chain.invoke(input_data)