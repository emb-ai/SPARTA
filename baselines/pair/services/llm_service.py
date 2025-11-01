import json, os, torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import uvicorn
import re

# bind to physical GPUs 0 and 1 only
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"#"0,1"#

MODEL_NAME = "Qwen/Qwen3-32B"#"nvidia/Llama-3.1-Nemotron-70B-Instruct-HF"#
tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME)
model      = AutoModelForCausalLM.from_pretrained(
                 MODEL_NAME,
                 torch_dtype="auto", #torch.bfloat16,
                 device_map="auto"
             )
model.eval()

# FORMAT:
# You MUST format your response in JSON with two elements: 'improvement' and 'adversarial_query'.
# - The 'improvement' value should contain a few sentences explaining your reasoning for the new query based on the previous IoU score.
# - The 'adversarial_query' value will contain the new, refined text query.

# YOUR RESPONSE:
# {
#   "improvement": "The IoU score is still very high, indicating the model easily understood the simple query. I will rephrase by adding a distracting clause about the dog's color and changing the sentence structure to be more complex.",
#   "adversarial_query": "Of the two animals present, identify the one that is brown and positioned on the right-hand side of the image."
# }

MAIN_PROMPT = """
You are a helpful red-teaming assistant specializing in testing the robustness of Vision-Language Models (VLMs). Your task is to generate adversarial text queries for a reasoning segmentation model. The model is designed to output a segmentation mask based on the query text and an image.

SETTING:
You will be given an original text query that correctly identifies an object in an image, resulting in a high Intersection over Union (IoU) score. Your goal is to iteratively rephrase this query. The rephrased query MUST be grammatically correct and semantically equivalent to the original, but it should cause the VLM to fail, resulting in a very low IoU score.

OBJECTIVE:
Your primary objective is to find a paraphrase that minimizes the IoU score. A score of 1.0 means your paraphrase failed (the VLM understood it perfectly). A score of 0.0 means your paraphrase was a perfect success (the VLM was completely confused).

STRATEGY:
To create effective adversarial paraphrases, you should remember:
1. Your paraphrases should be grammatically correct.
2. Your paraphrases should be semantically equivalent to the original query.
3. Do not add extra information that is not mentioned in the original query.
4. Your paraphrase must preserve the original query's capitalization pattern (e.g., initial uppercase vs. lowercase) and its terminal punctuation (e.g., period, question mark).

INPUT FROM USER:
In each turn, you will receive the original query, the previous adversarial query you generated and the resulting IoU score of the adversarial query from the VLM.

FORMAT:
You MUST format your response as only 'adversarial_query'.
- The 'adversarial_query' value will contain the new, refined text query.

USER INPUT:
{
  "original_query": "{orig_query}",
  "previous_query": "{previous_query}",
  "iou_score": {iou_score}
}
You MUST format your response only as 'adversarial_query'.
Do not add anything before and after it.
YOUR RESPONSE:
  "adversarial_query": 
"""

class Req(BaseModel):
    orig_query:     str
    previous_query: str
    iou:            float

class Resp(BaseModel):
    adversarial_query: str

app = FastAPI()

def chat(orig, prev, iou):
    user = json.dumps({"orig_query": orig,
                       "previous_query": prev,
                       "iou_score": iou}, ensure_ascii=False)
    return [
        {"role": "system", "content": MAIN_PROMPT},
        {"role": "user",   "content": user},
    ]

@app.post("/next_query", response_model=Resp)
@torch.no_grad()
def next_query(req: Req):
    # 1) build prompt
    messages = chat(req.orig_query ,req.previous_query, req.iou)
    inputs   = tokenizer.apply_chat_template(
                  messages, tokenize=True, add_generation_prompt=True,
                  return_tensors="pt", enable_thinking=False)

    # 2) generate
    ids  = model.generate(inputs.cuda(),
                          max_new_tokens=512,
                          pad_token_id=tokenizer.eos_token_id)
    txt  = tokenizer.decode(ids[0, inputs.size(1):],
                            skip_special_tokens=True).strip()

    # 3) sanitise: try a few patterns until we get the bare query
    # ------------------------------------------------------------
    # a) if it already looks like JSON, load & return
    try:
        obj = json.loads(txt)
        return {"adversarial_query": obj.get("adversarial_query", txt)}
    except Exception:
        pass

    # b) strip code-block fences ```json ... ```
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```",
                       txt, flags=re.S | re.I)
    if fenced:
        try:
            obj = json.loads(fenced.group(1))
            return {"adversarial_query": obj.get("adversarial_query", txt)}
        except Exception:
            txt = fenced.group(1)     # fall back to raw

    # c) remove leading label `adversarial_query:` if present
    m = re.search(r'adversarial_query"\s*:\s*"(.*)"', txt)
    if m:
        return {"adversarial_query": m.group(1).strip()}

    m = re.search(r'adversarial_query\s*:\s*(.*)', txt)
    if m:
        return {"adversarial_query": m.group(1).strip(' "\'')}

    # d) nothing matched → return the raw text
    return {"adversarial_query": txt}

if __name__ == "__main__":
    uvicorn.run(
        app,                        # ← pass the *object*, not "module:app"
        host="0.0.0.0",
        port=8001,
        log_level='info',
        reload=False,               # ← make sure no worker-respawn occurs
        workers=1                   # (default);  keep single process
    )