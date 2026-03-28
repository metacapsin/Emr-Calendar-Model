from fastapi import FastAPI

from api.routes.predict_slot import router as predict_router

app = FastAPI(title='Healthcare Slot Recommendation API')

app.include_router(predict_router)


@app.get('/health')
def health_check():
    return {'status': 'ok'}
