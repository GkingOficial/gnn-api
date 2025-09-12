import os
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi import HTTPException
from model.config_flags import FLAGS
from model.predictor import Predictor
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title = 'GNN Toxicity Predictor')

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_CKPT_PATH = "app/model/weights/ToxicidadeGCN_fold0.ckpt-159"

if not os.path.exists(MODEL_CKPT_PATH + ".index"):
    raise FileNotFoundError(f"Ficheiro de checkpoint do modelo não encontrado em {MODEL_CKPT_PATH}. Verifique se os ficheiros .ckpt, .index e .meta estão no diretório correto.")

predictor = Predictor(MODEL_CKPT_PATH, FLAGS)

class SMILESRequest(BaseModel):
    smiles: str

@app.get("/")
def read_root():
    return {"status": "API GNN online!"}

@app.post("/predict")
def predict(request: SMILESRequest):
    print(f"SMILES recebido: {request.smiles}")
    result = predictor.predict(request.smiles)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result