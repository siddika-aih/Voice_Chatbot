"""DCB Bank specific operations"""
from typing import Dict, Any
from retrieval.hybrid_search import hybrid_search

class BankTool:
    """DCB Bank knowledge retrieval and operations"""
    
    DECLARATION = {
        "name": "query_dcb_knowledge",
        "description": "Search DCB Bank's knowledge base for information about accounts, loans, credit cards, policies, interest rates, and banking services. Use this for DCB Bank specific queries.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Query about DCB Bank services or products"
                }
            },
            "required": ["query"]
        }
    }
    
    async def execute(self, query: str) -> Dict[str, Any]:
        """
        Query DCB Bank knowledge base
        
        Returns:
            Relevant information from DCB Bank documents
        """
        try:
            context = await hybrid_search.hybrid_retrieve(query, top_k=3)
            
            return {
                "success": True,
                "query": query,
                "information": context,
                "source": "DCB Bank Knowledge Base"
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to retrieve information: {str(e)}"
            }
