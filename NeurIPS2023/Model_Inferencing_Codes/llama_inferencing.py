import cv2
import llama
import torch
from PIL import Image
import os
import requests
import pandas as pd
from pycocotools.coco import COCO
import json
import skimage.io as io

device = "cuda" if torch.cuda.is_available() else "cpu"

annFile = r"/scratch/atherala/Hallucinations_MLLMs/captions_val2014.json"
coco=COCO(annFile)
questions_path = r"/scratch/atherala/Hallucinations_MLLMs/VQA_v2_Selected_15k.xlsx"
questions_df = pd.read_excel(questions_path)
llama_answers_list = []
ans_cnt = 0



llama_dir = "/scratch/atherala/Hallucinations_MLLMs/LLAMA/LLaMA-Adapter/llama_adapter_v2_multimodal7b/llama"

# choose from BIAS-7B, LORA-BIAS-7B, LORA-BIAS-7B-v21
model, preprocess = llama.load("BIAS-7B", llama_dir, llama_type="7B", device=device)
model.eval()



for row in questions_df.iterrows():
    try:
        question = row[1]["Questions"]
        image_id = row[1]["Image_Ids"]
        actual_answers = row[1]["Answers"]
        img = coco.loadImgs(int(image_id))
        image = io.imread(img[0]['coco_url'])
        img = Image.fromarray(image)
        prompt = llama.format_prompt(question)
        img = preprocess(img).unsqueeze(0).to(device)
        generated_text = model.generate(img, [prompt])[0]

        print("Question is :- ",question)
        print("Answer from LLAMA:-",generated_text)
        print("Actual Answers are:- ",actual_answers)
        if generated_text:
            ans_cnt+=1
        llama_answers_list.append(generated_text)
    except Exception as e:
        print("Exception Caught ",e)
        generated_text = "error in image"
        llama_answers_list.append(generated_text)
    


questions_df["LLAMA_Answers"] = llama_answers_list
print("No. of Answers Generated are:- ",ans_cnt)
questions_df.to_csv("llama_results_15k.csv")





