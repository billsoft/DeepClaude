from typing import List, Dict
from app.utils.logger import logger

class MessageProcessor:
    @staticmethod
    def convert_to_deepseek_format(messages: List[Dict]) -> List[Dict]:
        """转换消息格式为 DeepSeek 所需的格式
        
        主要功能：
        1. 合并连续的相同角色消息
        2. 在连续消息之间插入过渡消息
        3. 确保消息严格交替
        
        Args:
            messages: 原始消息列表
            
        Returns:
            处理后的消息列表
        """
        processed = []
        temp_content = []
        current_role = None

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            # 跳过空消息
            if not content:
                continue

            # 处理系统消息
            if role == "system":
                if processed and processed[0]["role"] == "system":
                    processed[0]["content"] += f"\n{content}"
                else:
                    processed.insert(0, {"role": "system", "content": content})
                continue

            # 合并连续的相同角色消息
            if role == current_role:
                temp_content.append(content)
            else:
                if temp_content:
                    processed.append({
                        "role": current_role,
                        "content": "\n".join(temp_content)
                    })
                temp_content = [content]
                current_role = role

        # 添加最后一组消息
        if temp_content:
            processed.append({
                "role": current_role,
                "content": "\n".join(temp_content)
            })

        # 确保消息交替
        final_messages = []
        for i, msg in enumerate(processed):
            if i > 0 and msg["role"] == final_messages[-1]["role"]:
                if msg["role"] == "user":
                    final_messages.append({"role": "assistant", "content": "请继续。"})
                else:
                    final_messages.append({"role": "user", "content": "请继续。"})
            final_messages.append(msg)

        logger.debug(f"转换后的消息格式: {final_messages}")
        return final_messages

    @staticmethod
    def validate_messages(messages: List[Dict]) -> bool:
        """验证消息格式是否有效
        
        验证规则：
        1. 消息列表不能为空
        2. 相邻消息的角色不能相同
        
        Args:
            messages: 消息列表
            
        Returns:
            bool: 格式是否有效
        """
        if not messages:
            return False
            
        for i in range(1, len(messages)):
            if messages[i]["role"] == messages[i-1]["role"]:
                return False
                
        return True