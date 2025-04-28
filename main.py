import os

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from pyngrok import conf, ngrok

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
from langchain.agents import AgentExecutor, create_react_agent
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pyngrok import conf, ngrok

from src.save_memory import (
    get_combined_history,
    save_chatbot_memory,
    save_workshop_memory,
)
from src.tools import (
    PainPointDetector,
    hr_tool,
    question_tool,
    strategy_tool,
    web_search_tool,
)

conf.get_default().auth_token = os.getenv('NGROK_API_KEY')
tools = [hr_tool, strategy_tool, web_search_tool, question_tool, PainPointDetector]

template = """
You are a smart assistant to facilitator, listening to conversations between a facilitator, HR, and Strategy team.
Your goal is to assist by using tools when necessary or providing a brief acknowledgement.
You are listening to {role} right now.
Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

You have access to the following tools:
{tools}

Chat history:
{chat_history}

New input:
{input}

Thought:{agent_scratchpad}
""" 

prompt = PromptTemplate(
    template=template,
    input_variables=[
        "tools",         # Tool descriptions
        "chat_history",  # Loaded from memory
        'role',
        "tool_names",       
        "input",         # The new user utterance
        "agent_scratchpad" # Intermediate reasoning
    ])

# Ensure the LLM used is standard, not the structured_llm
llm = ChatOpenAI(model='gpt-4o-mini-2024-07-18',temperature=0)

# Create the agent with the standard LLM
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

# Create the agent executor (as before)
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    handle_parsing_errors=True,
    verbose=True,
    max_iterations=3,
    return_intermediate_steps=True,
    memory=ConversationBufferMemory(memory_key="chat_history", input_key="input") # Ensure input_key matches
)

def revoke(role: str, user_txt:str):
    """
    Args:
        - role: can be 'facilitator', 'hr', or 'strategy'
        - user_text: transcribed text from their speech
    """
    # 1) Save the workshop message with timestamp
    save_workshop_memory(role, user_txt)

    # 2) Build combined memory payload
    # chat_history = get_combined_history()

    # 3) Run it
    try:
        response = agent_executor.invoke({
    # "tool_names": ", ".join([t.name for t in tools]),
    # "tools": "\n".join(f"- {t.name}: {t.description}" for t in tools),
    "chat_history": get_combined_history(),
    "input": user_txt,
    'role': role
})
        final_answer = response.get('output', '')
        intermediate_steps = response.get('intermediate_steps', [])

        tools_used_info = []
        if intermediate_steps:
            for action, observation in intermediate_steps:
                # action is an AgentAction object (or similar structure)
                tools_used_info.append({
                    "tool_used": action.tool,
                    "tool_input": action.tool_input,
                    "tool_response": observation
                })

        # Construct your desired final output structure
        custom_output = {
            "final_answer": final_answer,
            "action_taken": bool(intermediate_steps), # True if any tool was used
            "tools_sequence": tools_used_info # List of tools used in order
        }
        # 4) Log the chatbot’s reply
        save_chatbot_memory(user_txt, response['output'])
        return custom_output
        
    except Exception as e:
        print(e)

    

    return None


class QueryRequest(BaseModel):
    role: str
    text: str

# API endpoint
app = FastAPI()

@app.post("/ask")
async def ask_question(request: QueryRequest):

    role = request.role
    question = request.text  # Access the 'question' attribute from the model
    # Process the question and return the response
    response = revoke(role, question)  
    return response

# Create ngrok tunnel
public_url = ngrok.connect(8000)
print(f"🚀 Your public URL: {public_url}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)