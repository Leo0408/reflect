import argparse
import json
import os
import sys

# --- Path setup ---
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'main'))

# --- Function imports ---
from main.exp import * 
from main.execute_replan import run_correction
from main.data import load_data
from main.gen_data import *
from LLM.prompt import LLMPrompter

FOLDER_NAME = 'makeCoffee/makeCoffee-1' 
API_KEY = "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"

task_info = {
    "name": "make coffee", # name of the task
    "task_idx": 5, # index of the task, defined in TASK_LIST in constants.py
    "num_samples": 1, # the number of samples to generate, this applies for randomly injected failures (e.g. see "Task 6" in tasks.json, to automatically generate two dropping failures which occurred at different times)
    "failure_injection": False, # whether to inject failures manaully or automatically
    "folder_name": "makeCoffee-1", # name of the folder to save data
    "scene": "FloorPlan16", # scene id as in ai2thor
    "chosen_failure": "occupied", # selected failure type, can also be blocking, occupied_put, ambiguous_plan, wrong_perception, drop, failed_action, and missing_step. See tasks.json for examples.
    "gt_failure_reason": "The robot failed to put the mug inside the coffee machine because there was already a cup inside it, occupying the space.",
    "gt_failure_step": "00:51",
    "preactions": [ # actions taken to set object state before task execution
        "(dirty_obj, Mug)"
    ],
    "failure_injection_params" : { # parameters used to configure the environment according to chosen_failure
        "src_obj_type": "Cup",
        "target_obj_type": "CoffeeMachine",
        "disp_x": 0.0,
        "disp_z": 0.05,
        "disp_y": 0.02
    },
    "actions": [ # the original robot plan
        "(navigate_to_obj, Mug)",
        "(pick_up, Mug)",
        "(navigate_to_obj, Sink)",
        "(put_on, Mug, SinkBasin)",
        "(toggle_on, Faucet)",
        "(toggle_off, Faucet)",
        "(pick_up, Mug)",
        "(pour, Mug, Sink)",
        "(navigate_to_obj, CoffeeMachine)",
        "(put_in, Mug, CoffeeMachine)",
        "(toggle_on, CoffeeMachine)",
        "(toggle_off, CoffeeMachine)",
        "(pick_up, Mug)",
        "(put_on, Mug, CounterTop)"
    ],
    "success_condition": "a clean mug is filled with coffee and on top of the countertop."
}

def main():
    llm_prompter = LLMPrompter(gpt_version="gpt-3.5-turbo", api_key=API_KEY)

    # with open('main/tasks.json') as f:
    #     tasks = json.load(f)

    # def show_video(video_path, video_width=300):
    #     video_file = open(video_path, "r+b").read()
    #     video_url = f"data:video/mp4;base64,{b64encode(video_file).decode()}"
    #     return HTML(f"""<video width={video_width} controls><source src="{video_url}"></video>""")

    # run_data_gen(data_path=os.getcwd(), task=task_info)
    # FOLDER_NAME = 'makeCoffee/makeCoffee-1'  # specify the task folder name here. In this example, it's makeCoffee/makeCoffee-1.
    # show_video(f'thor_tasks/{FOLDER_NAME}/original-video.mp4')

    # WITH_AUDIO = 1 # 1: using audio deteceted with wav2clip, 0: using ground truth audio information
    # events, task, object_list, interact_actions, nav_actions = load_data(f"main/thor_tasks/{FOLDER_NAME}", task_info)
    # print(len(events))

    # Sensory-input summary
    # detected_sounds = []
    # if WITH_AUDIO == 1:
    #     detected_sounds = run_sound_module(FOLDER_NAME, object_list)
    # generate_scene_graphs(FOLDER_NAME, events, object_list, nav_actions, interact_actions, WITH_AUDIO, detected_sounds)
    with open(f'main/state_summary/{FOLDER_NAME}/global_sg.pkl', 'rb') as f:
        global_sg = pickle.load(f)
        print("================ Global SG ================")
        print(global_sg)

    # Event-based summary & Subgoal-based summary
    # generate_summary(FOLDER_NAME, events, nav_actions, interact_actions, WITH_AUDIO, detected_sounds)
    run_reasoning(FOLDER_NAME, llm_prompter, global_sg)

if __name__ == "__main__":
    main()