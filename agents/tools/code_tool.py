"""Code execution tool"""
from typing import Dict, Any
import sys
from io import StringIO
import contextlib

class CodeExecutionTool:
    """Execute Python code safely"""
    
    # Use Gemini's native code execution (preferred)
    DECLARATION = "code_execution"  # Built-in Gemini tool
    
    # Custom declaration if needed
    CUSTOM_DECLARATION = {
        "name": "execute_python",
        "description": "Execute Python code for calculations, data processing, or complex logic. Use when user asks to calculate, analyze data, or perform computations.",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute"
                }
            },
            "required": ["code"]
        }
    }
    
    async def execute(self, code: str) -> Dict[str, Any]:
        """
        Execute Python code in sandboxed environment
        
        Returns:
            Dict with execution result or error
        """
        try:
            # Capture stdout
            stdout = StringIO()
            
            # Restricted globals for safety
            safe_globals = {
                "__builtins__": {
                    "print": print,
                    "len": len,
                    "range": range,
                    "sum": sum,
                    "min": min,
                    "max": max,
                    "abs": abs,
                    "round": round,
                    "int": int,
                    "float": float,
                    "str": str,
                    "list": list,
                    "dict": dict,
                }
            }
            
            # Execute with output capture
            with contextlib.redirect_stdout(stdout):
                exec(code, safe_globals)
            
            output = stdout.getvalue()
            
            return {
                "success": True,
                "output": output if output else "Code executed successfully (no output)",
                "code": code
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Code execution failed: {str(e)}"
            }
