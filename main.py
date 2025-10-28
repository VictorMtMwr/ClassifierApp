from app import create_app
from app.config import HOST, PORT, DEBUG

app = create_app()

if __name__ == '__main__':
    app.run(host=HOST, port=PORT, debug=DEBUG)
