import os
os.system('pip install -r requirements.txt')

from app import app 

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=5100, reload=True)
