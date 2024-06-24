from typing import List
from fastapi import Depends, FastAPI, HTTPException
from model_dic import brand_model
import joblib


from starlette.middleware.cors import CORSMiddleware
from pydantic import BaseModel


class CarModel(BaseModel):
    model_id: int
    model_name: str
    brand_name: str


class BrandModel(BaseModel):
    brand_id: int
    models: List[CarModel]


class ModelRequest(BaseModel):
    Mileage: int
    Engine_volume: int
    Engine_power: int
    Registered: bool
    Year: int
    brand_enc: int
    body_enc: int
    fuel_type_enc: int
    model_enc: int


fuel_dict = {"Benzin": 0, "Dizel": 1, "Elektro": 2, "Hibrid": 3, "Plin": 4}
body_dict = {
    "Caddy": 0,
    "Kabriolet": 1,
    "Karavan": 2,
    "Kombi": 3,
    "Limuzina": 4,
    "Malo auto": 5,
    "Monovolumen": 6,
    "Off Road": 7,
    "Ostalo": 8,
    "Pick up": 9,
    "SUV": 10,
    "Sportski/kupe": 11,
    "Terenac": 12,
}

model = joblib.load("rf1_base_rf.pkl")
print(model.feature_names_in_)


def start_application():
    app = FastAPI()
    origins = ["*"]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    return app


app = start_application()


@app.get("/models", response_model=List[BrandModel])
async def get_models():
    brand_models = []
    for brand_id, models in brand_model.items():
        brand_models.append(
            BrandModel(
                brand_id=int(brand_id), models=[CarModel(**model) for model in models]
            )
        )
    return brand_models


@app.get("/fuel_types")
async def get_fuels():
    return fuel_dict


@app.get("/body_types")
async def get_body_types():
    return body_dict


@app.post("/post_car")
async def test_car(request: ModelRequest):
    niz = [
        request.Mileage,
        request.Engine_volume,
        request.Engine_power,
        request.Registered,
        request.Year,
        request.brand_enc,
        request.body_enc,
        request.fuel_type_enc,
        request.model_enc,
    ]
    try:
        k = model.predict([niz])[0]
        return {"prediction": k}
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@app.get("/")
async def index():
    return {"ping": "Pong"}
