from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from llava.eval.run_llava import eval_model
import torch
import os
from PIL import Image
import requests
import pandas as pd
from pycocotools.coco import COCO
import json
import skimage.io as io

model_path = "liuhaotian/llava-v1.5-13b"

tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=get_model_name_from_path(model_path)
)
annFile = r"/scratch/atherala/Hallucinations_MLLMs/captions_val2014.json"
coco=COCO(annFile)
questions_path = r"/scratch/atherala/Hallucinations_MLLMs/VQA_v2_Selected_15k.xlsx"
questions_df = pd.read_excel(questions_path)
llava_answers_list = []
ans_cnt = 0
device = "cuda"

model.to(device)
for row in questions_df.iterrows():
    try:
        question = row[1]["Questions"]
        image_id = row[1]["Image_Ids"]
        actual_answer = row[1]["Ground_Truth_Answers"]
        img = coco.loadImgs(int(image_id))
        image = io.imread(img[0]['coco_url'])
        llava_prompt = question
        input_ids = tokenizer_image_token(llava_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(device)
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().to(device),
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                num_beams=3,
                # no_repeat_ngram_size=3,
                max_new_tokens=1024,
                use_cache=True)
                
        input_token_len = input_ids.shape[1]
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        generated_text = outputs.strip()
        print("Question is :- ",question)
        print("Answer from LLAVA:-",generated_text)
        print("Actual Answers are:- ",actual_answer)
        llava_answers_list.append(generated_text)
        if generated_text:
            ans_cnt+=1
    except Exception as e:
        print("Exception Caught ",e)
        generated_text = "error in image"
        llava_answers_list.append(generated_text)

questions_df["LLAVA_Answers"] = llava_answers_list
print("No. of Answers Generated are:- ",ans_cnt)
questions_df.to_csv("llava_results_15k.csv")
        