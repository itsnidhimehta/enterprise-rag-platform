from fastapi import FastAPI
app = FastAPI(title="Enterprise RAG API")

@app.get("/health")
def health():
    return {"status": "ok"}