from fastapi import FastAPI
from api.router import router
import uvicorn
import sys

# Initialize FastAPI app
app = FastAPI()

# Include the router
app.include_router(router)


def main() -> None:
    """
    Main function to run the FastAPI application.
    """
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    sys.exit(main())
