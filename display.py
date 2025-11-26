import json
from IPython.display import HTML
from base64 import b64encode

from main.gen_data import *
from main.data import load_data
from main.exp import *
from main.execute_replan import run_correction
from LLM.prompt import LLMPrompter
from main import *

# You may change the GPT version here
llm_prompter = LLMPrompter(gpt_version="gpt-4-0613", api_key=API_KEY)

with open('tasks.json') as f:
    tasks = json.load(f)

def show_video(video_path, video_width=300):
  video_file = open(video_path, "r+b").read()
  video_url = f"data:video/mp4;base64,{b64encode(video_file).decode()}"
  return HTML(f"""<video width={video_width} controls><source src="{video_url}"></video>""")