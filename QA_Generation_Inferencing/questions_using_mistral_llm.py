from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import json
import random

device = "cuda" # the device to load the model onto
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1",device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

json_fp = open(r"/home/atherala/NLP_Project/captions_val2014.json")
coco_captions = json.load(json_fp)
coco_captions_sub = random.sample(coco_captions["annotations"],100)


final_qs_list = []
final_ans_list = []
final_captions_list = []
final_img_ids_list = []

for kv_dict in coco_captions_sub:
    image_id = kv_dict["image_id"]
    caption = kv_dict["caption"]
    questions_list = []
    answers_list = []
    answers_list = []
    captions_list = []
    img_ids_list = []
    print("The Caption is :-",caption)
    prompt = "Given the caption : " +caption+" create strictly no more than 2 questions per caption and their corresponding answers which can be answered from the caption. Please generate only 2 questions per caption and Please Provide output in the form of Q: and A:"
    model_inputs = tokenizer([prompt], return_tensors="pt").to(device)
    model.to(device)
    generated_ids = model.generate(**model_inputs, max_new_tokens=1000, do_sample=True,temperature = 0.2,top_p = 0.5,pad_token_id=tokenizer.eos_token_id)
    tokens_list = tokenizer.batch_decode(generated_ids,skip_special_tokens = True)[0].split("\n")
    qa_list = []
    for qa in tokens_list:
        if qa.startswith("Q:") or qa.startswith("A:"):
            qa_list.append(qa)
    if not (len(qa_list)%2 == 0):
        qa_list = qa_list[:len(qa_list)-1]
        
    for sent in qa_list:
        
        if sent.startswith("Q:"):
            questions_list.append(sent)
            captions_list.append(caption)
            img_ids_list.append(image_id)
        elif sent.startswith("A:"):
            answers_list.append(sent)
    if len(questions_list) >= 2 and len(answers_list) >= 2 and len(captions_list) >= 2 and len(img_ids_list) >= 2:
        final_qs_list.extend(questions_list[:2])
        final_ans_list.extend(answers_list[:2])
        final_captions_list.extend(captions_list[:2])
        final_img_ids_list.extend(img_ids_list[:2])
    print("Final Questions: ",final_qs_list)
    print("Final Answers: ",final_ans_list)

            



questions_df = pd.DataFrame()
questions_df["Mistral_Questions"] = final_qs_list
questions_df["Mistral_Answers"] = final_ans_list
questions_df["Captions"] = final_captions_list
questions_df["Image_Ids"] = final_img_ids_list
questions_df.drop_duplicates(inplace=True)
questions_df.to_csv("questions_df.csv")