from fastapi import FastAPI, Request
from pydantic import BaseModel
import uvicorn
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "FastKASSIM"))

from fkassim import FastKassim as fk

app = FastAPI()

class TextPair(BaseModel):
    text1: str
    text2: str

# 初期化（LTK: Label Tree Kernel）
FastKassim = fk.FastKassim(fk.FastKassim.LTK)

@app.post("/similarity")
def get_similarity(pair: TextPair):
    score = FastKassim.compute_similarity(pair.text1, pair.text2)
    return {"score": score}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
