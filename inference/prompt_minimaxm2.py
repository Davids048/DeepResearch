# MiniMax-M2 Tool Calling Configuration
# This file defines tool schemas for MiniMax-M2 model
# The tools are defined in vLLM-compatible format (OpenAI tool schema)
# vLLM will automatically convert these to MiniMax-M2's XML format

TOOLS_MINIMAXM2 = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Perform Google web searches then returns a string of the top search results. Accepts multiple queries.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "description": "The search query."
                        },
                        "minItems": 1,
                        "description": "The list of search queries."
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "visit",
            "description": "Visit webpage(s) and return the summary of the content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "The URL(s) of the webpage(s) to visit. Can be a single URL or an array of URLs."
                    },
                    "goal": {
                        "type": "string",
                        "description": "The specific information goal for visiting webpage(s)."
                    }
                },
                "required": ["url", "goal"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "PythonInterpreter",
            "description": "Executes Python code in a sandboxed environment. IMPORTANT: Any output you want to see MUST be printed to standard output using the print() function.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The Python code to execute. Remember to use print() statements for any output you want to see."
                    }
                },
                "required": ["code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "google_scholar",
            "description": "Leverage Google Scholar to retrieve relevant information from academic publications. Accepts multiple queries. This tool will also return results from google search",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "description": "The search query."
                        },
                        "minItems": 1,
                        "description": "The list of search queries for Google Scholar."
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "parse_file",
            "description": "This is a tool that can be used to parse multiple user uploaded local files such as PDF, DOCX, PPTX, TXT, CSV, XLSX, DOC, ZIP, MP4, MP3.",
            "parameters": {
                "type": "object",
                "properties": {
                    "files": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "The file name of the user uploaded local files to be parsed."
                    }
                },
                "required": ["files"]
            }
        }
    }
]
