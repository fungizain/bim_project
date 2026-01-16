import os
import uvicorn


def main():
    uvicorn.run("src.app:app", host="0.0.0.0", port=8000, reload=True)


def dev_main():
    os.environ["APP_ENV"] = "dev"
    main()


def prod_main():
    os.environ["APP_ENV"] = "prod"
    main()
