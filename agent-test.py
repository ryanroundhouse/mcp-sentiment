import gradio as gr
import ollama
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from mcp.client.stdio import StdioServerParameters
from smolagents import ToolCollection, CodeAgent
from smolagents.mcp_client import MCPClient

@dataclass
class ChatMessage:
    content: str
    role: str = "assistant"

class OllamaModel:
    def __init__(self, model_name: str = "qwen2.5-coder:14b"):
        self.model_name = model_name
        self.system_prompt = """
You are an AI assistant that can analyze text sentiment using the ryanroundhouse_mcp_sentimentpredict tool.
When given text to analyze, you should:
1. Use the ryanroundhouse_mcp_sentimentpredict tool to analyze the sentiment.
2. Call the tool using this exact format:
   <tool>ryanroundhouse_mcp_sentimentpredict</tool>
   <args>
   {"text": "your text here"}
   </args>

Example usage:
User: "I love this product, it's amazing!"
Assistant: Let me analyze the sentiment of this text.
<tool>ryanroundhouse_mcp_sentimentpredict</tool>
<args>
{"text": "I love this product, it's amazing!"}
</args>
"""

    def generate(self, messages: List[Dict[str, str]], **kwargs) -> ChatMessage:
        # Add system prompt at the beginning
        full_messages = [{"role": "system", "content": self.system_prompt}] + messages
        
        # Convert messages to Ollama format
        prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in full_messages])
        
        # Call Ollama
        response = ollama.generate(
            model=self.model_name,
            prompt=prompt,
            stream=False
        )
        
        return ChatMessage(content=response['response'])

# Initialize MCP client
mcp_client = MCPClient(
    {"url": "https://ryanroundhouse-mcp-sentiment.hf.space/gradio_api/mcp/sse"}
)

# Get and print available tools
tools = mcp_client.get_tools()
print("Available MCP tools:", [tool.name for tool in tools])

model = OllamaModel()  # Using qwen2.5-coder:14b by default
agent = CodeAgent(tools=[*tools], model=model)

def call_agent(message, history):
    print(f"Processing message: {message}")
    result = agent.run(message)
    print(f"Agent result: {result}")
    return str(result)

demo = gr.ChatInterface(
    fn=call_agent,
    type="messages",
    examples=["Write some text to analyze"],
    title="Agent with MCP Tools",
    description="This is a simple agent that uses MCP tools to answer questions.",
)

demo.launch()