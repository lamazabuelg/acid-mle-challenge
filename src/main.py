import os
import uvicorn
import logging
import settings
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from endpoints.files import files_router
from endpoints.models import models_router

# create logger
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)

# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

# create formatter
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# add formatter to ch
ch.setFormatter(formatter)

# add ch to logger
logger.addHandler(ch)

app = FastAPI(
    title="Machine Learning Engineer Challenge - Luis Ángel Mazabuel García",
    description="Service to manage Data Scientist's models. Builded by Luis Ángel Mazabuel García (Data Scientist & Machine Learning Engineer) for the Machine Learning Engineer Challenge of ACID Labs.",
    version=settings.APP_VERSION,
)


app.include_router(files_router)
app.include_router(models_router)


@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse("/docs")


if __name__ == "__main__":
    uvicorn.run(
        app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), log_level="info"
    )
