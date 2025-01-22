import json
import traceback
import requests
import base64

from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI
from model_configurations import get_model_configuration


gpt_chat_version = 'gpt-4o'
gpt_config = get_model_configuration(gpt_chat_version)

model = AzureChatOpenAI(
        model=gpt_config['model_name'],
        deployment_name=gpt_config['deployment_name'],
        openai_api_key=gpt_config['api_key'],
        openai_api_version=gpt_config['api_version'],
        azure_endpoint=gpt_config['api_base'],
        temperature=gpt_config['temperature']
        )

json_format = '{{"Result": [{{ "date": "yyyy-MM-dd", "name": "節日" }}, {{ "date": "yyyy-MM-dd", "name": "節日" }}] }}'

store = {}

def get_history_session_id(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

@tool
def get_month_holidays(country, year, month) -> json:
    '''取得某個國家某個月份的假日資訊。country是國家代碼,year是年,month是月。'''
    url = "https://calendarific.com/api/v2/holidays"
    params={
        "api_key": "XN2yuSI5LjPnY9b33Xqm1muqwl8dj744",
        "country": country,
        "year": year,
        "month": month,
        "language": "zh"
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        try:
            holidays = response.json()['response']['holidays']
            return {
                "Result": [
                    {
                        "date": holiday['date']['iso'],
                        "name": holiday['name']
                    }
                    for holiday in holidays
                ]
            }
        except Exception as e:
            print(f"Error: {e}\n{response.json()}")
    else:
        print(f"Request error：{response.status_code}\n{response.text}")
    return json.loads('{"Result": {}}')

def generate_hw01(question):
    prompt = """
        答案請用此 JSON 格式呈現, 格式如下：
        {{
            "Result": [
                {{
                    "date": "yyyy-MM-dd",
                    "name": "節日名稱"
                }}
            ]
        }}
        """
    messages = [
        SystemMessage(content = [{"type": "text", "text": prompt},]),
        HumanMessage(content = [{"type": "text", "text": question},]),
    ]
    response = model.invoke(messages)
    result = response.content.strip().removeprefix("```json").removesuffix("```")
    return result;

def generate_hw02(question):
    prompt = ChatPromptTemplate.from_messages([
        ('system', '取得某個國家某個月份的假日資訊'),
        ('system', f'回答的所有節日，用繁體中文回答節日名稱，用以下JSON格式呈現:{json_format}'), 
        ('human', '{input}'),
        ('human', '{agent_scratchpad}'),
    ])
    tool_function = [get_month_holidays]
    response = AgentExecutor(
        agent=create_tool_calling_agent(model, tool_function, prompt),
        tools=tool_function,
        verbose=True
    ).invoke({'input': question})
    result = JsonOutputParser().parse(response['output'])
    return json.dumps(result, indent=4, ensure_ascii=False)
    
def generate_hw03(question2, question3):
    prompt = ChatPromptTemplate.from_messages([
        ('system', '取得某個國家某月份的假日資訊'),
        ('placeholder', '{history}'),
        ('human', '{input}'),
        ('human', '{agent_scratchpad}'),
    ])
    tool_function = [get_month_holidays]
    session_id = "get_holiday"
    config = {'configurable': {'session_id': session_id}}
    agent = create_tool_calling_agent(model, tool_function, prompt)
    agent_executor = AgentExecutor(agent = agent, tools = tool_function)
    agent_with_history = RunnableWithMessageHistory(
        agent_executor,
        input_messages_key='input',
        history_messages_key='history',
        get_session_history=get_history_session_id
    )
    response = agent_with_history.invoke(
        {'input': f'{question2}\n回答的所有節日，用繁體中文回答節日名稱，用以下JSON格式呈現:{json_format}'},
        config=config
    )
    response = agent_with_history.invoke(
        {'input': f'{question3}\n回答是否需要將節日新增到節日清單中。根據問題判斷該節日是否存在於清單中，如果不存在，則為 true；否則為 false，並說明做出此回答的理由，用以下JSON格式呈現:{{ "Result": {{ "add": true, "reason": "理由描述" }} }}'},
        config=config
    )
    jason_parser = JsonOutputParser()
    result = jason_parser.invoke(response['output'])
    return json.dumps(result, indent=4, ensure_ascii=False)

def generate_hw04(question):
    with open("baseball.png", "rb") as png_file:
        png_data = base64.b64encode(png_file.read()).decode("utf-8")
    prompt = """
    回答圖片中的分數，只須回答一個分數，並回答用JSON格式呈現:{{ "Result": {{ "score": 5498 }} }}
    """
    messages = [
        SystemMessage(content = [{"type": "text", "text": prompt},]),
        HumanMessage([{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{png_data}"},},]),
    ]
    response = model.invoke(messages)
    jason_parser = JsonOutputParser()
    result = jason_parser.invoke(response)
    return json.dumps(result, indent=4, ensure_ascii=False)

def demo(question):
    llm = AzureChatOpenAI(
            model=gpt_config['model_name'],
            deployment_name=gpt_config['deployment_name'],
            openai_api_key=gpt_config['api_key'],
            openai_api_version=gpt_config['api_version'],
            azure_endpoint=gpt_config['api_base'],
            temperature=gpt_config['temperature']
    )
    message = HumanMessage(
            content=[
                {"type": "text", "text": question},
            ]
    )
    response = llm.invoke([message])
    
    return response

