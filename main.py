import uvicorn
from app import create_app
from app.config import HOST, PORT, DEBUG

app = create_app()

if __name__ == '__main__':
    uvicorn.run(app, host=HOST, port=PORT, log_level="debug" if DEBUG else "info")
