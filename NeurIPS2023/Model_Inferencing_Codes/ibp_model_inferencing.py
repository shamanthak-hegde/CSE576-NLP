from transformers import AutoModelForCausalLM, AutoTokenizer,InstructBlipProcessor, InstructBlipForConditionalGeneration
import torch
import os
from PIL import Image
import requests
import pandas as pd
from pycocotools.coco import COCO
import json
import skimage.io as io

annFile = r"/scratch/atherala/Hallucinations_MLLMs/captions_val2014.json"
coco=COCO(annFile)
# json_fp = open(r"/home/atherala/NLP_Project/captions_val2014.json")
# coco_captions = json.load(json_fp)
questions_path = r"/scratch/atherala/Hallucinations_MLLMs/VQA_v2_selected_15k.csv"
questions_df = pd.read_csv(questions_path)
ibp_answers_list = []
ans_cnt = 0
device = "cuda"
ibp_model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-vicuna-7b")
ibp_processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b")
ibp_model.to(device)
for row in questions_df.iterrows():
    try:
        question = row[1]["Questions"]
        image_id = row[1]["Image_Ids"]
        actual_answers = row[1]["Answers"]
        img = coco.loadImgs(int(image_id))
        image = io.imread(img[0]['coco_url'])
        
        # ibp_prompt = "Please Answer the question based upon the image provided. The question is "+question
        ibp_prompt = question
        inputs = ibp_processor(images=image, text=ibp_prompt, return_tensors="pt").to(device)
        outputs = ibp_model.generate(
            **inputs,
            do_sample=False,
            num_beams=5,
            max_length=256,
            min_length=1,
            top_p=0.9,
            repetition_penalty=1.5,
            length_penalty = 1.0,
            temperature=1,
        )
        generated_text = ibp_processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
        print("Question is :- ",question)
        print("Answer from IBP:-",generated_text)
        print("ACtual Answers are:- ",actual_answers)
        if generated_text:
            ans_cnt+=1
        ibp_answers_list.append(generated_text)
    except Exception as e:
        print("Exception Caught ",e)
        generated_text = "error in image"
        ibp_answers_list.append(generated_text)
    


questions_df["IBP_Answers"] = ibp_answers_list
print("No. of Answers Generated are:- ",ans_cnt)
questions_df.to_csv("ibp_results_15k.csv")
