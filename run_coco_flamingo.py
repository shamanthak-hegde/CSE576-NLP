import json
from PIL import Image
import torch
from open_flamingo import create_model_and_transforms
from huggingface_hub import hf_hub_download
import os
from tqdm import tqdm

if __name__=="__main__":
    model, image_processor, tokenizer = create_model_and_transforms(
        clip_vision_encoder_path="ViT-L-14",
        clip_vision_encoder_pretrained="openai",
        lang_encoder_path="anas-awadalla/mpt-1b-redpajama-200b",
        tokenizer_path="anas-awadalla/mpt-1b-redpajama-200b",
        cross_attn_every_n_layers=1
    )

    checkpoint_path = hf_hub_download("openflamingo/OpenFlamingo-3B-vitl-mpt1b", "checkpoint.pt")
    model.load_state_dict(torch.load(checkpoint_path), strict=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device: ", device)
    data_dir = "/home/shegde23/val2014/"
    model.to(device)
    os.system("nvidia-smi")
    f = json.load(open('QA_Dataset.json'))

    image1 = Image.open(data_dir+f[0]['file_name']).convert("RGB")
    image2 = Image.open(data_dir+f[1]['file_name']).convert("RGB")

    question1 = f[0]['questions'][0]
    question2 = f[1]['questions'][0]

    answer1 = f[0]['answers'][0]
    answer2 = f[1]['answers'][0]

    for i in tqdm(range(2,len(f))):
        image = Image.open(data_dir+f[i]['file_name']).convert("RGB")

        vision_x = [image_processor(image1).unsqueeze(0), image_processor(image2).unsqueeze(0), image_processor(image).unsqueeze(0)]
        vision_x = torch.cat(vision_x, dim=0)
        vision_x = vision_x.unsqueeze(1).unsqueeze(0)

        answers = []
        for j in range(len(f[i]['questions'])):
            prompt = f[i]['questions'][j]
            tokenizer.padding_side = "left" # For generation padding tokens should be on the left
            lang_x = tokenizer(
            [f"<image>Question: {question1}? Answer: {answer1}<|endofchunk|><image>Question: {question2}? Answer: {answer2}<|endofchunk|><image>Question: {prompt} Answer: "],
            return_tensors="pt",
            )
            generated_text = model.generate(
                vision_x=vision_x.to(device),
                lang_x=lang_x["input_ids"].to(device),
                attention_mask=lang_x["attention_mask"].to(device),
                max_new_tokens=20,
                num_beams=3,
            )

            answer = tokenizer.decode(generated_text[0])
            # get the answer without the examples
            answer = answer.split('Question: ')[-1]
            if len(answer.split('Answer:')) > 1:
                answer = answer.split('Answer:')[-1].replace('<|endofchunk|>', '').replace('.', '')
            answers.append(answer)
        
        f[i]['pred_answers'] = answers
        if(i==(5000-1) or i==(10000-1) or i==(15000-1)):
            x=json.dumps(f)
            with open('pred_flam_cocoqa_intmd.json','w') as w:
                w.write(x)

    x = json.dumps(f)
    with open('pred_flam_cocoqa.json','w') as w:
        w.write(x)
            