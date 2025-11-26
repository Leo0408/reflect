#!/usr/bin/env python3
"""
Real-World Robot Failure Detection Implementation Example
基于Reflect论文的实际场景实现示例
"""

import os
import json
import numpy as np
import cv2
import open3d as o3d
from PIL import Image
import torch
from typing import Dict, List, Tuple, Optional

class RealWorldFailureDetector:
    """实际场景机器人失败检测器"""
    
    def __init__(self, config_path: str):
        """
        初始化失败检测器
        
        Args:
            config_path: 配置文件路径
        """
        self.config = self.load_config(config_path)
        self.detector = None
        self.llm_prompter = None
        self.setup_models()
    
    def load_config(self, config_path: str) -> Dict:
        """加载配置文件"""
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def setup_models(self):
        """初始化模型"""
        # 初始化物体检测器
        from mdetr_object_detector import mdetr_efficientnetB3_phrasecut
        self.detector = mdetr_efficientnetB3_phrasecut(pretrained=True)
        self.detector.eval()
        
        # 初始化LLM接口
        from LLM.prompt import LLMPrompter
        self.llm_prompter = LLMPrompter(
            gpt_version="gpt-4",
            api_key=self.config.get("openai_api_key")
        )
    
    def process_robot_execution(self, data_path: str, task_config: Dict) -> Dict:
        """
        处理机器人执行数据并检测失败
        
        Args:
            data_path: 数据路径
            task_config: 任务配置
            
        Returns:
            失败检测结果
        """
        print(f"开始处理任务: {task_config['name']}")
        
        # 1. 加载数据
        rgb_data, depth_data, robot_states = self.load_robot_data(data_path)
        
        # 2. 生成场景图
        scene_graphs = self.generate_scene_graphs(
            rgb_data, depth_data, robot_states, task_config
        )
        
        # 3. 生成分层总结
        L1_summary = self.generate_L1_summary(scene_graphs, robot_states)
        L2_summary = self.generate_L2_summary(L1_summary, robot_states)
        
        # 4. LLM推理
        failure_result = self.run_failure_reasoning(
            L1_summary, L2_summary, task_config
        )
        
        return failure_result
    
    def load_robot_data(self, data_path: str) -> Tuple[List, List, Dict]:
        """加载机器人执行数据"""
        # 这里需要根据实际数据格式实现
        # 示例：从zarr文件加载数据
        import zarr
        
        meta_data = zarr.open(f"{data_path}/replay_buffer.zarr", 'r')
        
        # 加载RGB和深度数据
        rgb_data = []
        depth_data = []
        
        total_frames = len(meta_data['data/stage'])
        for i in range(total_frames):
            rgb = self.load_image(f"{data_path}/videos/color/{i}.0.0.0")
            depth = self.load_depth(f"{data_path}/videos/depth/{i}.0.0")
            rgb_data.append(rgb)
            depth_data.append(depth)
        
        # 加载机器人状态
        robot_states = {
            'gripper_pos': meta_data['data/gripper_pos'][:],
            'stage': meta_data['data/stage'][:]
        }
        
        return rgb_data, depth_data, robot_states
    
    def load_image(self, image_path: str) -> np.ndarray:
        """加载图像"""
        from imagecodecs import imread
        return imread(image_path)
    
    def load_depth(self, depth_path: str) -> np.ndarray:
        """加载深度图"""
        from imagecodecs import imread
        return imread(depth_path)
    
    def generate_scene_graphs(self, rgb_data: List, depth_data: List, 
                            robot_states: Dict, task_config: Dict) -> List:
        """生成场景图序列"""
        from real_world_get_local_sg import get_scene_graph
        from real_world_scene_graph import SceneGraph
        
        scene_graphs = []
        total_points_dict = {}
        bbox3d_dict = {}
        
        # 获取关键帧
        key_frames = self.get_key_frames(robot_states)
        
        for step_idx in key_frames:
            print(f"处理第 {step_idx} 帧")
            
            rgb = rgb_data[step_idx]
            depth = depth_data[step_idx]
            
            # 生成场景图
            local_sg, bbox3d_dict, total_points_dict, bbox2d_dict = get_scene_graph(
                args=None,
                rgb=rgb,
                depth=depth,
                step_idx=step_idx,
                object_list=task_config['object_list'],
                distractor_list=task_config.get('distractor_list', []),
                detector=self.detector,
                total_points_dict=total_points_dict,
                bbox3d_dict=bbox3d_dict,
                meta_data={'data/gripper_pos': robot_states['gripper_pos']},
                task_info=task_config
            )
            
            scene_graphs.append({
                'step_idx': step_idx,
                'scene_graph': local_sg,
                'bbox2d': bbox2d_dict
            })
        
        return scene_graphs
    
    def get_key_frames(self, robot_states: Dict) -> List[int]:
        """获取关键帧"""
        # 基于动作变化选择关键帧
        stages = robot_states['stage']
        key_frames = [0]  # 第一帧
        
        prev_stage = stages[0]
        for i, curr_stage in enumerate(stages[1:], 1):
            if curr_stage != prev_stage:
                key_frames.append(i)
            prev_stage = curr_stage
        
        return key_frames
    
    def generate_L1_summary(self, scene_graphs: List, robot_states: Dict) -> List[str]:
        """生成L1总结"""
        from real_world_hierarchical_prompt import get_scene_text
        
        L1_captions = []
        actions = self.config.get('actions', [])
        
        for sg_data in scene_graphs:
            step_idx = sg_data['step_idx']
            scene_graph = sg_data['scene_graph']
            
            # 获取动作
            stage = robot_states['stage'][step_idx]
            if stage < len(actions):
                action = actions[stage]
            else:
                action = "Unknown"
            
            # 获取场景描述
            scene_text = get_scene_text(scene_graph)
            
            # 生成时间戳
            timestep = self.convert_step_to_timestep(step_idx)
            
            # 组合L1描述
            caption = f"{timestep}. Action: {action}. Visual observation: {scene_text}"
            L1_captions.append(caption)
        
        return L1_captions
    
    def generate_L2_summary(self, L1_captions: List[str], robot_states: Dict) -> List[str]:
        """生成L2总结"""
        L2_captions = []
        
        # 基于动作结束帧生成L2描述
        stages = robot_states['stage']
        action_end_frames = []
        
        prev_stage = stages[0]
        for i, curr_stage in enumerate(stages[1:], 1):
            if curr_stage != prev_stage:
                action_end_frames.append(i-1)
            prev_stage = curr_stage
        
        # 为每个动作结束帧生成L2描述
        for end_frame in action_end_frames:
            timestep = self.convert_step_to_timestep(end_frame)
            for caption in L1_captions:
                if timestep in caption:
                    L2_caption = caption.replace("Action", "Goal")
                    L2_captions.append(L2_caption)
                    break
        
        return L2_captions
    
    def run_failure_reasoning(self, L1_captions: List[str], L2_captions: List[str], 
                            task_config: Dict) -> Dict:
        """运行失败推理"""
        from real_world_hierarchical_prompt import run_reasoning
        
        # 构建参数
        class Args:
            def __init__(self):
                self.folder_name = task_config['general_folder_name']
                self.ablation_type = 0
                self.audio_ver = 1
        
        args = Args()
        
        # 创建全局场景图
        global_sg = self.create_global_scene_graph(L1_captions)
        
        # 运行推理
        reasoning_result = run_reasoning(args, task_config, global_sg)
        
        return reasoning_result
    
    def create_global_scene_graph(self, L1_captions: List[str]):
        """创建全局场景图"""
        from real_world_scene_graph import SceneGraph
        # 这里需要根据L1描述重建全局场景图
        # 简化实现
        return SceneGraph()
    
    def convert_step_to_timestep(self, step_idx: int, fps: int = 30) -> str:
        """将帧索引转换为时间戳"""
        seconds = step_idx / fps
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        return f"{minutes:02d}:{seconds:02d}"


def main():
    """主函数示例"""
    # 配置文件示例
    config = {
        "openai_api_key": "your-api-key-here",
        "actions": [
            "Pick up cup",
            "Put cup in coffee machine", 
            "Toggle on coffee machine",
            "Pick up cup",
            "Put cup on table"
        ]
    }
    
    # 任务配置示例
    task_config = {
        "name": "make coffee",
        "general_folder_name": "makeCoffee2",
        "object_list": [
            "coffee machine",
            "purple cup", 
            "blue cup with handle",
            "table on the left of sink"
        ],
        "actions": [
            "Pick up cup",
            "Put cup in coffee machine",
            "Toggle on coffee machine", 
            "Pick up cup",
            "Put cup on table"
        ],
        "success_condition": "a cup filled with coffee is on table",
        "gt_failure_reason": "A mug is already inside the coffee machine, as a result, the cup cannot be put inside."
    }
    
    # 初始化检测器
    detector = RealWorldFailureDetector("config.json")
    
    # 处理机器人执行数据
    data_path = "/path/to/robot/data"
    result = detector.process_robot_execution(data_path, task_config)
    
    print("失败检测结果:")
    print(f"预测失败原因: {result.get('pred_failure_reason', 'N/A')}")
    print(f"预测失败步骤: {result.get('pred_failure_step', 'N/A')}")
    print(f"真实失败原因: {result.get('gt_failure_reason', 'N/A')}")


if __name__ == "__main__":
    main()
