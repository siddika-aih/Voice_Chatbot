"""Google Search tool using Gemini's native grounding"""
import os
from typing import Dict, Any

class GoogleSearchTool:
    """Google Search via Gemini grounding"""
    
    # Native tool - no execution needed, Gemini handles it
    DECLARATION = "google_search_retrieval"  # Built-in Gemini tool
    
    # For custom search (if needed)
    CUSTOM_DECLARATION = {
        "name": "google_search",
        "description": "Search Google for current information, news, facts, or real-time data. Use when user asks about current events, latest news, or things beyond DCB Bank knowledge.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query"
                }
            },
            "required": ["query"]
        }
    }
    
    async def execute(self, query: str) -> Dict[str, Any]:
        """
        Search Google (if using custom implementation)
        Note: Gemini's native google_search_retrieval is preferred
        """
        try:
            # This would use Google Custom Search API
            import aiohttp
            
            api_key = os.getenv("GOOGLE_SEARCH_API_KEY")
            cx = os.getenv("GOOGLE_SEARCH_CX")
            
            if not api_key or not cx:
                return {
                    "success": False,
                    "message": "Google Search API not configured. Using Gemini's native search instead."
                }
            
            url = "https://www.googleapis.com/customsearch/v1"
            params = {
                "key": api_key,
                "cx": cx,
                "q": query,
                "num": 3
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    data = await response.json()
                    
                    results = []
                    if "items" in data:
                        for item in data["items"][:3]:
                            results.append({
                                "title": item.get("title"),
                                "snippet": item.get("snippet"),
                                "link": item.get("link")
                            })
                    
                    return {
                        "success": True,
                        "query": query,
                        "results": results
                    }
                    
        except Exception as e:
            return {
                "success": False,
                "message": f"Search failed: {str(e)}"
            }
