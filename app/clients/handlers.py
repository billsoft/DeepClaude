import logging
import json
from typing import List, Dict

# 配置日志
logger = logging.getLogger(__name__)

def validate_and_convert_tools(tools: List[Dict], target_format: str = 'claude-3') -> List[Dict]:
    """验证并转换工具格式

    Args:
        tools: 工具定义列表
        target_format: 目标格式，默认claude-3

    Returns:
        List[Dict]: 转换后的工具列表
    """
    if not tools or not isinstance(tools, list):
        logger.warning("无效的工具列表")
        return []

    logger.info(f"开始验证和转换 {len(tools)} 个工具至 {target_format} 格式")
    valid_tools = []

    for i, tool in enumerate(tools):
        if not isinstance(tool, dict):
            logger.warning(f"工具[{i}]不是字典格式，跳过")
            continue

        try:
            # 处理不同的工具格式
            
            # 处理OpenAI格式（包含function字段的工具）
            if "function" in tool:
                logger.info(f"验证工具[{i}]: {tool.get('function', {}).get('name', '未命名')} (OpenAI格式 -> {target_format})")
                
                function_data = tool["function"]
                name = function_data.get("name", "未命名工具")
                description = function_data.get("description", "")
                parameters = function_data.get("parameters", {})
                
                if target_format == 'claude-3':
                    # 转换为Claude格式
                    claude_tool = {
                        "name": name,
                        "description": description,
                        "input_schema": parameters
                    }
                    valid_tools.append(claude_tool)
                    logger.info(f"已将OpenAI格式工具转为Claude格式: {name}")
                else:
                    # 保持OpenAI格式
                    valid_tools.append(tool)
                    logger.info(f"保留OpenAI格式: {name}")
                
                continue
            
            # 处理Claude格式（包含type:custom的工具）
            if "type" in tool and tool["type"] == "custom":
                logger.info(f"验证工具[{i}]: {tool.get('name', '未命名')} (Claude格式 -> {target_format})")
                
                name = tool.get("name", "未命名工具")
                description = tool.get("description", "")
                
                # 获取工具模式
                schema = None
                for schema_field in ["tool_schema", "input_schema"]:
                    if schema_field in tool:
                        schema = tool[schema_field]
                        break
                
                if not schema:
                    logger.warning(f"工具[{i}]缺少schema定义: {name}")
                    continue
                
                if target_format == 'claude-3':
                    # 已经是Claude格式，只需标准化
                    claude_tool = {
                        "name": name,
                        "description": description,
                        "input_schema": schema
                    }
                    valid_tools.append(claude_tool)
                    logger.info(f"已标准化Claude格式工具: {name}")
                else:
                    # 转换为OpenAI格式
                    openai_tool = {
                        "type": "function",
                        "function": {
                            "name": name,
                            "description": description,
                            "parameters": schema
                        }
                    }
                    valid_tools.append(openai_tool)
                    logger.info(f"已将Claude格式工具转为OpenAI格式: {name}")
                
                continue
            
            # 处理简化格式（只包含name、description和parameters的工具）
            if set(["name", "description"]).issubset(set(tool.keys())):
                logger.info(f"验证工具[{i}]: {tool.get('name', '未命名')} (简化格式 -> {target_format})")
                
                name = tool.get("name", "未命名工具")
                description = tool.get("description", "")
                
                # 获取参数定义，支持多种可能的字段名
                parameters = None
                for param_field in ["parameters", "schema", "input_schema", "tool_schema"]:
                    if param_field in tool:
                        parameters = tool[param_field]
                        break
                
                if not parameters:
                    logger.warning(f"工具[{i}]缺少参数定义: {name}")
                    parameters = {"type": "object", "properties": {}}
                
                if target_format == 'claude-3':
                    # 转换为Claude格式
                    claude_tool = {
                        "name": name,
                        "description": description,
                        "input_schema": parameters
                    }
                    valid_tools.append(claude_tool)
                    logger.info(f"已将简化格式工具转为Claude格式: {name}")
                else:
                    # 转换为OpenAI格式
                    openai_tool = {
                        "type": "function",
                        "function": {
                            "name": name,
                            "description": description,
                            "parameters": parameters
                        }
                    }
                    valid_tools.append(openai_tool)
                    logger.info(f"已将简化格式工具转为OpenAI格式: {name}")
                
                continue
            
            # 处理没有匹配任何已知格式的工具
            logger.warning(f"工具[{i}]格式未知: {json.dumps(tool)[:100]}...")
            
        except Exception as e:
            logger.error(f"验证工具[{i}]时出错: {e}", exc_info=True)
    
    logger.info(f"工具验证完成: {len(tools)} 个输入工具 -> {len(valid_tools)} 个有效工具")
    return valid_tools 