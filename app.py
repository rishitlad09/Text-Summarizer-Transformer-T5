from fastapi import FastAPI,Request
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration,T5Tokenizer
import torch
import re
from fastapi.templating import Jinja2Templates # UI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from transformers import MarianMTModel,MarianTokenizer

# initialize our fastapi application

app=FastAPI(title="Text Summarizer App",description="Text Summarization using T5",version="1.0")


#model & tokenizer
model=T5ForConditionalGeneration.from_pretrained("./saved_summary_model")
tokenizer=T5Tokenizer.from_pretrained("./saved_summary_model")

#Translator model & tokenizer
trans_model_name = "Helsinki-NLP/opus-mt-en-hi"
translate_tokenizer = MarianTokenizer.from_pretrained(trans_model_name)

# device
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model.to(device)
translate_model = MarianMTModel.from_pretrained(trans_model_name).to(device)
#templating

templates=Jinja2Templates(directory="templates")


#Input schema for dialogue => string

class DialogueInput(BaseModel):
    dialogue: str
    
class TranslateInput(BaseModel):
    text: str


def clean_data(text):
    text=re.sub(r"\r\n"," ",text) #lines
    text=re.sub(r"\s+"," ",text) #spaces
    text=re.sub(r"<.*?>"," ",text)#html tags
    text=text.strip().lower()
    return text

def summarize_dialogue(dialogue: str) -> str:
    dialogue=clean_data(dialogue) # clean
    # tokenize
    inputs=tokenizer(
        dialogue,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt"
    ).to(device)
    # generate the summary => (token ids)
    model.to(device)
    target = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        min_length=50,
        max_length=150,
        num_beams=4,
        early_stopping=True,
    )
    #Convert token ids to text(summary) => decoding 
    summary=tokenizer.decode(target[0],skip_special_tokens=True) # Skip EOS,SEP
    return summary 

# API Endpoints
@app.post("/summarize/")
async def summarize(dialogue_input: DialogueInput):
    summary=summarize_dialogue(dialogue_input.dialogue)
    return {"summary":summary}





@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request, 
        name="index.html", 
        context={},
    )



















