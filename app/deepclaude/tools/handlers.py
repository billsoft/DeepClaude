from typing import Dict, List, Any, Optional
from app.utils.logger import logger
from .validators import ToolValidator
from .converters import ToolConverter
import uuid
import json
import copy

class ToolHandler:
    """工具处理器类，用于验证、转换和处理工具调用"""
    
    def __init__(self):
        """初始化工具处理器"""
        pass
        
    def validate_and_convert_tools(self, tools: List[Dict], target_format: str = 'claude-3') -> List[Dict]:
        """
        验证并转换工具格式，确保其符合目标API要求
        
        Args:
            tools: 工具定义列表
            target_format: 目标格式，可选 'claude-3' 或 'openai'
            
        Returns:
            List[Dict]: 验证和转换后的工具列表，如果无有效工具则返回None
        """
        if not tools or not isinstance(tools, list) or len(tools) == 0:
            logger.warning("未提供有效的工具列表")
            return None
            
        validated_tools = []
        input_tools_count = len(tools)
        logger.info(f"开始验证和转换 {input_tools_count} 个工具至 {target_format} 格式")
        
        for i, tool in enumerate(tools):
            try:
                if not isinstance(tool, dict):
                    logger.warning(f"工具[{i}]不是字典类型，跳过: {tool}")
                    continue
                    
                logger.debug(f"处理工具[{i}]: {json.dumps(tool, ensure_ascii=False)[:100]}...")
                
                # 检测OpenAI函数格式
                if "function" in tool or (tool.get("type") == "function" and "function" in tool):
                    # 获取函数定义
                    function = tool.get("function", tool) if tool.get("type") == "function" else tool["function"]
                    
                    if not isinstance(function, dict):
                        logger.warning(f"工具[{i}]函数结构不是字典: {function}")
                        continue
                        
                    if "name" not in function:
                        logger.warning(f"工具[{i}]函数缺少name字段: {function}")
                        continue
                        
                    # 获取核心属性
                    name = function.get("name", f"未命名工具_{i}")
                    description = function.get("description", "")
                    parameters = function.get("parameters", {"type": "object", "properties": {}})
                    
                    # 确保parameters有效
                    if not isinstance(parameters, dict):
                        logger.warning(f"工具[{i}] {name} 的parameters不是字典: {parameters}")
                        parameters = {"type": "object", "properties": {}}
                    
                    # 确保type字段存在且为object
                    if "type" not in parameters or parameters["type"] != "object":
                        logger.debug(f"工具[{i}] {name} 的parameters.type设置为object (原值: {parameters.get('type')})")
                        parameters["type"] = "object"
                    
                    # 确保properties存在
                    if "properties" not in parameters or not isinstance(parameters["properties"], dict):
                        logger.debug(f"工具[{i}] {name} 的parameters缺少properties字段或不是字典")
                        parameters["properties"] = {}
                    
                    # 根据目标格式创建工具
                    if target_format == 'claude-3':
                        validated_tool = {
                            "name": name,
                            "description": description,
                            "input_schema": {
                                "type": "object",
                                "properties": parameters.get("properties", {}),
                                "required": parameters.get("required", [])
                            }
                        }
                    else:  # openai格式
                        validated_tool = {
                            "type": "function",
                            "function": {
                                "name": name,
                                "description": description,
                                "parameters": parameters
                            }
                        }
                        
                    validated_tools.append(validated_tool)
                    logger.info(f"转换工具[{i}]: {name} (OpenAI格式 -> {target_format})")
                    
                # 检测Claude格式工具
                elif "name" in tool and ("input_schema" in tool or "description" in tool):
                    name = tool["name"]
                    description = tool.get("description", "")
                    
                    if target_format == 'claude-3':
                        # 如果已经是Claude格式，只需确保包含必要字段
                        validated_tool = {
                            "name": name,
                            "description": description
                        }
                        
                        # 处理input_schema
                        if "input_schema" in tool and isinstance(tool["input_schema"], dict):
                            # 克隆input_schema
                            input_schema = copy.deepcopy(tool["input_schema"])
                            
                            # 确保type字段存在
                            if "type" not in input_schema:
                                input_schema["type"] = "object"
                                
                            # 确保properties字段存在
                            if "properties" not in input_schema or not isinstance(input_schema["properties"], dict):
                                input_schema["properties"] = {}
                                
                            validated_tool["input_schema"] = input_schema
                        else:
                            # 创建默认input_schema
                            validated_tool["input_schema"] = {
                                "type": "object",
                                "properties": {}
                            }
                            
                            # 如果有custom字段，尝试从中提取属性
                            if "custom" in tool and isinstance(tool["custom"], dict):
                                custom = tool["custom"]
                                if "properties" in custom and isinstance(custom["properties"], dict):
                                    validated_tool["input_schema"]["properties"] = custom["properties"]
                                if "required" in custom and isinstance(custom["required"], list):
                                    validated_tool["input_schema"]["required"] = custom["required"]
                        
                        validated_tools.append(validated_tool)
                        logger.info(f"验证工具[{i}]: {name} (Claude格式 -> {target_format})")
                    else:
                        # 转换为OpenAI格式
                        parameters = {"type": "object", "properties": {}}
                        
                        # 从input_schema提取属性
                        if "input_schema" in tool and isinstance(tool["input_schema"], dict):
                            input_schema = tool["input_schema"]
                            if "properties" in input_schema and isinstance(input_schema["properties"], dict):
                                parameters["properties"] = input_schema["properties"]
                            if "required" in input_schema and isinstance(input_schema["required"], list):
                                parameters["required"] = input_schema["required"]
                                
                        validated_tool = {
                            "type": "function",
                            "function": {
                                "name": name,
                                "description": description,
                                "parameters": parameters
                            }
                        }
                        
                        validated_tools.append(validated_tool)
                        logger.info(f"转换工具[{i}]: {name} (Claude格式 -> {target_format})")
                
                # 尝试处理其他格式
                elif "name" in tool and ("properties" in tool or "parameters" in tool):
                    name = tool["name"]
                    description = tool.get("description", "")
                    
                    # 处理parameters
                    parameters = {}
                    if "parameters" in tool and isinstance(tool["parameters"], dict):
                        parameters = tool["parameters"]
                    elif "properties" in tool and isinstance(tool["properties"], dict):
                        parameters = {"type": "object", "properties": tool["properties"]}
                        if "required" in tool and isinstance(tool["required"], list):
                            parameters["required"] = tool["required"]
                    
                    # 根据目标格式创建工具
                    if target_format == 'claude-3':
                        validated_tool = {
                            "name": name,
                            "description": description,
                            "input_schema": {
                                "type": "object",
                                "properties": parameters.get("properties", {}),
                                "required": parameters.get("required", [])
                            }
                        }
                    else:
                        validated_tool = {
                            "type": "function",
                            "function": {
                                "name": name,
                                "description": description,
                                "parameters": parameters
                            }
                        }
                        
                    validated_tools.append(validated_tool)
                    logger.info(f"转换工具[{i}]: {name} (通用格式 -> {target_format})")
                else:
                    logger.warning(f"工具[{i}]不是已知格式，跳过: {json.dumps(tool, ensure_ascii=False)[:100]}...")
                    continue
            except Exception as e:
                logger.error(f"处理工具[{i}]时出错: {e}")
                continue
        
        logger.info(f"工具验证完成: {input_tools_count} 个输入工具 -> {len(validated_tools)} 个有效工具")
        return validated_tools if validated_tools else None
        
    def _is_valid_function_tool(self, tool: Dict) -> bool:
        """验证Claude格式的函数工具是否有效"""
        if not isinstance(tool, dict) or "function" not in tool:
            return False
            
        function = tool["function"]
        if not isinstance(function, dict):
            return False
            
        # 检查必需字段
        required_fields = ["name", "description", "parameters"]
        for field in required_fields:
            if field not in function:
                return False
                
        # 验证参数结构
        parameters = function["parameters"]
        if not isinstance(parameters, dict):
            return False
            
        # 确保parameters至少包含基本结构
        if "type" not in parameters:
            return False
            
        if parameters["type"] != "object":
            return False
            
        # 检查属性定义
        if "properties" in parameters and not isinstance(parameters["properties"], dict):
            return False
            
        return True
        
    def _is_valid_custom_tool(self, tool: Dict) -> bool:
        """验证自定义工具是否有效"""
        if not isinstance(tool, dict):
            return False
            
        # 验证必需字段
        required_fields = ["name", "description", "input_schema"]
        for field in required_fields:
            if field not in tool:
                return False
                
        # 验证input_schema字段结构
        input_schema = tool.get("input_schema")
        if not isinstance(input_schema, dict):
            return False
            
        # 验证input_schema结构
        if "type" not in input_schema or input_schema["type"] != "object":
            return False
            
        # 检查properties字段
        if "properties" not in input_schema or not isinstance(input_schema["properties"], dict):
            return False
            
        return True
        
    def _convert_openai_to_claude(self, openai_tool: Dict) -> Dict:
        """将OpenAI格式的工具转换为Claude格式
        
        Args:
            openai_tool: OpenAI格式的工具定义
            
        Returns:
            Claude格式的工具定义
        """
        try:
            if "function" in openai_tool:
                function = openai_tool["function"]
            elif "type" in openai_tool and openai_tool["type"] == "function":
                function = openai_tool
            else:
                logger.warning(f"无法识别的OpenAI工具格式: {openai_tool}")
                return None
                
            # 提取函数名称和描述
            name = function.get("name", "未命名工具")
            description = function.get("description", "")
            
            # 提取参数
            parameters = function.get("parameters", {})
            properties = parameters.get("properties", {})
            required = parameters.get("required", [])
            
            # 创建Claude格式的工具
            claude_tool = {
                "name": name,
                "description": description,
                "input_schema": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            }
            
            logger.info(f"转换OpenAI工具 '{name}' 为Claude格式: {json.dumps(claude_tool, ensure_ascii=False)}")
            return claude_tool
        except Exception as e:
            logger.error(f"转换OpenAI工具到Claude格式失败: {e}")
            return None
        
    def format_tool_call_for_streaming(self, tool_call_data: Dict, chat_id: str, created_time: int) -> Dict:
        """将工具调用格式化为流式输出格式"""
        try:
            response = {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "created": created_time,
                "model": "deepclaude",
                "choices": [{
                    "index": 0,
                    "delta": {
                        "tool_calls": [tool_call_data]
                    }
                }],
                "finish_reason": None
            }
            return response
        except Exception as e:
            logger.error(f"格式化工具调用失败: {e}")
            return {
                "error": {
                    "message": str(e),
                    "type": "internal_error"
                }
            }
    
    async def process_tool_call(self, tool_call: Dict, **kwargs) -> Dict:
        """处理工具调用，实现抽象方法"""
        # 这里简单地返回一个示例响应，实际实现可能会调用外部服务或执行特定功能
        logger.info(f"处理工具调用: {tool_call.get('function', {}).get('name', '未知工具')}")
        return {
            "status": "success",
            "result": "工具调用结果示例"
        }

    def _final_validate_claude_tools(self, tools: List[Dict]) -> List[Dict]:
        """最终验证Claude工具格式，并修复任何问题
        
        Args:
            tools: 要验证的工具列表
            
        Returns:
            修复后的工具列表
        """
        if not tools:
            return []
            
        from .validators import ToolValidator
        
        valid_tools = []
        for i, tool in enumerate(tools):
            # 创建工具副本
            fixed_tool = {}
            
            # 确保基本字段存在
            for field in ["name", "description"]:
                if field in tool:
                    fixed_tool[field] = tool[field]
                else:
                    fixed_tool[field] = f"未命名工具_{i}" if field == "name" else ""
            
            # 确保type字段正确
            if "type" not in tool:
                fixed_tool["type"] = "custom"
            elif tool["type"] not in ["custom", "bash_20250124", "text_editor_20250124"]:
                logger.warning(f"工具[{i}]类型'{tool['type']}'不被Claude支持，修改为'custom'")
                fixed_tool["type"] = "custom"
            else:
                fixed_tool["type"] = tool["type"]
            
            # 处理tool_schema
            if fixed_tool["type"] == "custom":
                if "tool_schema" in tool and isinstance(tool["tool_schema"], dict):
                    schema = tool["tool_schema"].copy()
                    # 确保schema具有正确的type
                    if "type" not in schema:
                        schema["type"] = "object"
                    elif schema["type"] == "custom":
                        schema["type"] = "object"
                    
                    # 确保properties存在
                    if "properties" not in schema:
                        schema["properties"] = {}
                        
                    fixed_tool["tool_schema"] = schema
                else:
                    # 创建基本schema
                    fixed_tool["tool_schema"] = {"type": "object", "properties": {}}
            
            # 复制其他有效字段
            for key, value in tool.items():
                if key not in ["type", "name", "description", "tool_schema", "custom"]:
                    fixed_tool[key] = value
            
            # 最终验证
            is_valid, errors = ToolValidator.validate_claude_tool(fixed_tool)
            if not is_valid:
                logger.warning(f"工具[{i}]经过修复后仍有问题: {', '.join(errors)}")
                # 尝试修复剩余问题
                if "custom" in fixed_tool:
                    fixed_tool.pop("custom", None)
            
            valid_tools.append(fixed_tool)
            logger.debug(f"工具[{i}]最终格式: {json.dumps(fixed_tool, ensure_ascii=False)}")
            
        return valid_tools 