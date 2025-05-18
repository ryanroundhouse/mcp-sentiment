import gradio as gr
import ollama
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import json

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
You are an AI assistant that can use external tools to help answer user questions.

- If you need to know what tools are available, output ONLY <list_tools/> on a single line, with no explanation or extra text.
- When you want to use a tool, output ONLY <use_tool name=\"TOOL_NAME\" args='JSON_ARGS'/> on a single line, with no explanation or extra text.
- When you receive an Observation, use it to answer the user's question.

Do NOT explain your actions. Only output the special tags when you want to use a tool or list tools.

Example:
User: What tools can you use?
Assistant: <list_tools/>

User: Analyze the sentiment of "I love this!"
Assistant: <use_tool name=\"ryanroundhouse_mcp_sentimentpredict\" args='{"text": "I love this!"}'/>
"""

    def generate(self, messages: List[Dict[str, str]], **kwargs) -> ChatMessage:
        full_messages = [{"role": "system", "content": self.system_prompt}] + messages
        prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in full_messages])
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
tools = mcp_client.get_tools()

# Build a tool info dictionary for quick lookup
TOOL_INFO = {tool.name: tool for tool in tools}

def get_tools_description():
    desc = []
    for tool in tools:
        args = getattr(tool, 'args', None)
        args_str = json.dumps(args) if args else "(no args)"
        desc.append(f"- {tool.name}: {getattr(tool, 'description', '')} Args: {args_str}")
    return "\n".join(desc)

def call_agent(message, history):
    conversation = []
    for user, assistant in history:
        conversation.append({"role": "user", "content": user})
        if assistant:
            conversation.append({"role": "assistant", "content": assistant})
    conversation.append({"role": "user", "content": message})

    step = 1
    while True:
        print(f"\n--- Step {step} ---")
        llm_response = agent.model.generate(conversation).content.strip()
        print(f"LLM response: {llm_response}")
        if llm_response == "<list_tools/>":
            tool_list = get_tools_description()
            print(f"Agent: Listing tools...\n{tool_list}")
            conversation.append({"role": "assistant", "content": tool_list})
            step += 1
            continue
        elif llm_response.startswith("<use_tool"):
            import re
            match = re.match(r'<use_tool name=\"([\w_]+)\" args=\'(.*?)\'\s*/>', llm_response)
            if match:
                tool_name, args_json = match.groups()
                args = json.loads(args_json)
                tool = TOOL_INFO.get(tool_name)
                if tool:
                    print(f"Agent: Calling tool {tool_name} with args {args}")
                    result = tool(**args)
                    observation = f"Observation: {result}"
                    print(f"Agent: Observation: {result}")
                    conversation.append({"role": "assistant", "content": observation})
                    step += 1
                    continue
                else:
                    print(f"Agent: Tool {tool_name} not found.")
                    conversation.append({"role": "assistant", "content": f"Tool {tool_name} not found."})
                    step += 1
                    continue
            else:
                print("Agent: Malformed tool usage request.")
                conversation.append({"role": "assistant", "content": "Malformed tool usage request."})
                step += 1
                continue
        else:
            print(f"Agent: Final response: {llm_response}")
            return llm_response

model = OllamaModel()  # Using qwen2.5-coder:14b by default
agent = CodeAgent(tools=[*tools], model=model)

demo = gr.ChatInterface(
    fn=call_agent,
    type="messages",
    examples=["Write some text to analyze"],
    title="Agent with MCP Tools",
    description="This is a simple agent that uses MCP tools to answer questions.",
)

demo.launch()