# Workshop Facilitator AI Agent
We narrow down to a specific scenario where HR and strategy team attend a workshop on 'how to integrate AI into restructuring'. There are five tools being used in agent.
1. hr_tool: a RAG tool which will be grounded on any ingested domain knowledge/corporate policies/documents. folder path is 'data/HR'.
2. strategy_tool: a RAG tool which will be grounded on any ingested domain knowledge/corporate policies/documents. folder path is 'data/Strategy'.
3. web_search_tool: a public web search tool
4. question_tool: a question suggestion tool that can be called when any participant isn't active for a while.
5. PainPointDetector: as pain points are the key information we aim to extract from a workshop, we created such a tool specifically to capture this information.

Meanwhile, we store two conversation flows into `logs/'. One is the agent input & output, another is the workshop chats. With these information available, we can generate a post-workshop summary as well as feed all context information to our agent.
# How to use
run `python main.py`. then a local url/ngrok url will be exposed. can send request following this format   
`curl -X POST {URL exposed}/ask   -H "Content-Type: application/json"   -d "{\"role\":\"hr\",\"text\":\"how to lay off 10% of employees\"}"`, 'role' and 'text' are the two required arguments.