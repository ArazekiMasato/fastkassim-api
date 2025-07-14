from fastapi import FastAPI
from pydantic import BaseModel
import spacy
from FastKASSIM import FastKASSIM

app = FastAPI()
nlp = spacy.load("en_core_web_sm")
kassim = FastKASSIM(nlp)

class InputData(BaseModel):
    sentence1: str
    sentence2: str

@app.post("/syntax_score")
def syntax_score(data: InputData):
    score = kassim.similarity(data.sentence1, data.sentence2)
    return {"syntax_score": score}
