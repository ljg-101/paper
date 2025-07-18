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


class ReviewChain:
    """
    学术审稿链。
    """
    def __init__(self):
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_template(
            """
            你是一个面向SCI期刊的专业编辑，具有丰富的审稿经验以及论文编写经验
            请对以下内容{content}进行审查并提出专业性建议：  
            回答的格式为：
            [先对原文内容总结，再给出专业性建议]
            总字数在100字以内，且建议控制在两条以内，切忌不能凭空编造建议，如果没有需要提升的地方可以不用回答
            """
        )
        self.chain = self.prompt | self.llm

    def invoke(self, input_data):
        """
        使用提供的输入数据调用链以生成审稿结果。
        参数:
            input_data (dict): 包含 'history' 和 'question' 键的字典。
        返回:
            str: 审稿总结与建议。
        """
        return self.chain.invoke(input_data)