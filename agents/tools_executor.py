"""Tool execution coordinator"""
import asyncio
from typing import Dict, Any, List
from google.genai import types

from agents.tools.email_tool import EmailTool
from agents.tools.search_tool import GoogleSearchTool
from agents.tools.code_tool import CodeExecutionTool
from agents.tools.bank_tool import BankTool

class ToolExecutor:
    """Executes tools based on Gemini function calls"""
    
    def __init__(self):
        # Initialize all tools
        self.tools = {
            "send_email": EmailTool(),
            "google_search": GoogleSearchTool(),
            "execute_python": CodeExecutionTool(),
            "query_dcb_knowledge": BankTool()
        }
        
        print("🔧 Loaded tools:", list(self.tools.keys()))
    
    def get_tool_declarations(self) -> List[types.FunctionDeclaration]:
        """Get all tool declarations for Gemini"""
        declarations = []
        
        # Custom tools
        for name, tool in self.tools.items():
            if hasattr(tool, 'DECLARATION') and isinstance(tool.DECLARATION, dict):
                declarations.append(
                    types.FunctionDeclaration(
                        name=tool.DECLARATION["name"],
                        description=tool.DECLARATION["description"],
                        parameters=tool.DECLARATION["parameters"]
                    )
                )
        
        return declarations
    
    def get_native_tools(self) -> List[str]:
        """Get Gemini's native tools (google_search, code_execution)"""
        return ["google_search_retrieval", "code_execution"]
    
    async def execute_tool(self, function_call: types.FunctionCall) -> Dict[str, Any]:
        """
        Execute a tool based on function call from Gemini
        
        Args:
            function_call: FunctionCall object from Gemini
            
        Returns:
            Tool execution result
        """
        tool_name = function_call.name
        args = function_call.args
        
        print(f"\n🔧 Executing tool: {tool_name}")
        print(f"   Args: {args}")
        
        # Get tool instance
        if tool_name not in self.tools:
            return {
                "success": False,
                "error": f"Tool '{tool_name}' not found"
            }
        
        tool = self.tools[tool_name]
        
        try:
            # Execute tool with arguments
            result = await tool.execute(**args)
            print(f"✅ Tool result: {result.get('message', 'Success')}")
            return result
            
        except Exception as e:
            error_msg = f"Tool execution failed: {str(e)}"
            print(f"❌ {error_msg}")
            return {
                "success": False,
                "error": error_msg
            }

# Global instance
tool_executor = ToolExecutor()
