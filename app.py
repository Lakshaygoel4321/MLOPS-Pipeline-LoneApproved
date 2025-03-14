
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.responses import HTMLResponse, RedirectResponse
from uvicorn import run as app_run

from typing import Optional

from src.constants import APP_HOST, APP_PORT
from src.pipline.prediction_pipeline import USvisaData, USvisaClassifier
from src.pipline.training_pipeline import TrainPipeline

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory='templates')

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



class DataForm:
    def __init__(self, request: Request):
        self.request: Request = request
        self.Gender: Optional[str] = None
        self.Married: Optional[str] = None
        self.Dependents: Optional[str] = None
        self.Education: Optional[str] = None
        self.Self_Employed: Optional[str] = None
        self.ApplicantIncome: Optional[str] = None
        self.LoanAmount: Optional[str] = None
        self.Credit_History: Optional[str] = None
        self.Property_Area: Optional[str] = None
        
        

    async def get_usvisa_data(self):
        form = await self.request.form()
        self.Gender = form.get("Gender")
        self.Married = form.get("Married")
        self.Dependents = form.get("Dependents")
        self.Education = form.get("requires_job_training")
        self.Self_Employed = form.get("no_of_employees")
        self.ApplicantIncome = form.get("company_age")
        self.LoanAmount = form.get("region_of_employment")
        self.Credit_History = form.get("prevailing_wage")
        self.Property_Area = form.get("Property_Area")
        

@app.get("/", tags=["authentication"])
async def index(request: Request):

    return templates.TemplateResponse(
            "usvisa.html",{"request": request, "context": "Rendering"})


@app.get("/train")
async def trainRouteClient():
    try:
        train_pipeline = TrainPipeline()

        train_pipeline.run_pipeline()

        return Response("Training successful !!")

    except Exception as e:
        return Response(f"Error Occurred! {e}")


@app.post("/")
async def predictRouteClient(request: Request):
    try:
        form = DataForm(request)
        await form.get_usvisa_data()
        
        usvisa_data = USvisaData(
                                Gender= form.Gender,
                                Married = form.Married,
                                Dependents = form.Dependents,
                                Education = form.Education,
                                Self_Employed= form.Self_Employed,
                                ApplicantIncome= form.ApplicantIncome,
                                LoanAmount = form.LoanAmount,
                                Credit_History= form.Credit_History,
                                Property_Area= form.Property_Area,
                                
                                )
        
        usvisa_df = usvisa_data.get_usvisa_input_data_frame()

        model_predictor = USvisaClassifier()

        value = model_predictor.predict(dataframe=usvisa_df)[0]

        status = None
        if value == 1:
            status = "Loan is Approved"
        else:
            status = "Loan is Not-Approved"

        return templates.TemplateResponse(
            "usvisa.html",
            {"request": request, "context": status},
        )
        
    except Exception as e:
        return {"status": False, "error": f"{e}"}


if __name__ == "__main__":
    app_run(app, host=APP_HOST, port=APP_PORT)