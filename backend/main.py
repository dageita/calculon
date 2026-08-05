import argparse
import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.api import api_router
from app.config import settings
from app.logging_config import LOG_LEVELS, build_log_config, configure_logging, normalize_level


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.getLogger(__name__).info("llm training calculator started")
    logging.getLogger(__name__).info("To stop the server, press Ctrl+C")
    yield


def get_application():
    app = FastAPI(
        title=settings.PROJECT_NAME,
        openapi_url=f"{settings.API_V1_STR}/openapi.json",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_router, prefix="/llm_training_calculator")

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", help="The port to run the llm training calculator", default=8000)
    parser.add_argument(
        "--log-level",
        default="info",
        choices=LOG_LEVELS,
        help=(
            "Logging verbosity (default: info). "
            "warning/error/critical also silence LLMFlowSimulator stdout; "
            "info shows Python milestones only. "
            "Use debug for Calculon layer-by-layer dumps and simulator cout."
        ),
    )
    args = parser.parse_args()
    port = int(args.port)
    log_level = normalize_level(args.log_level)

    configure_logging(log_level)

    app = get_application()
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_config=build_log_config(log_level),
    )
