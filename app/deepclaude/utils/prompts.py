"""提供各种提示词模板"""

class PromptTemplates:
    """提示词模板集合"""
    
    @staticmethod
    def reasoning_prompt(question: str) -> str:
        """生成推理提示模板"""
        return f"""请思考下面这个问题，给出详细的分析过程：

{question}

分析思路：
"""

    @staticmethod
    def tool_decision_prompt(question: str, reasoning: str, tools_description: str) -> str:
        """生成工具决策提示模板"""
        return f"""用户问题：{question}

我的思考过程：
{reasoning}

可用工具：
{tools_description}

1. 仔细分析用户问题和思考过程。
2. 判断是否需要使用工具来回答问题。
3. 如果需要使用工具，请使用最合适的工具并提供所有必要的参数。
4. 如果不需要使用工具，直接回答用户问题。"""

    @staticmethod
    def final_answer_prompt(question: str, reasoning: str, tool_results: str = None) -> str:
        """生成最终回答提示模板"""
        tool_part = f"\n\n工具调用结果：\n{tool_results}" if tool_results else ""
        
        return f"""用户问题：{question}

我的思考过程：
{reasoning}{tool_part}

请根据以上信息，给出清晰、准确、有帮助的回答。不要在回答中提及你的思考过程或工具调用细节，直接回答用户问题。""" 