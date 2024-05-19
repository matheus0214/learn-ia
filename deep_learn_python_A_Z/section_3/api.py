from typing import List

from fastapi import FastAPI

from . import breast_cancer_carregar

app = FastAPI()


@app.post("/")
def index(data: List[float]):
    response = breast_cancer_carregar.classificar_um_registro(data)
    if response < 0.5:
        return {"response": "Beningno"}

    return {"response": "Maligno"}
