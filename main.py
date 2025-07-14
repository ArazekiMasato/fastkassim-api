# main.py

from fastapi import FastAPI
from pydantic import BaseModel
import sys
import os

# FastKASSIMモジュールのパスを追加（相対パス）
sys.path.append(os.path.join(os.path.dirname(__file__), "FastKASSIM"))

import fkassim.FastKassim as fkassim

app = FastAPI()

# FastKASSIMの初期化
fastkassim = fkassim.FastKassim(fkassim.FastKassim.LTK)

class InputData(BaseModel):
    text1: str
    text2: str

@app.post("/syntax_score")
async def syntax_score(data: InputData):
    score = fastkassim.compute_similarity(data.text1, data.text2)
    return {"score": score}
