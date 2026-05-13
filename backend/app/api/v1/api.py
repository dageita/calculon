from fastapi import APIRouter
from . import agent, benchmark, calculator

api_router = APIRouter()
api_router.include_router(calculator.router, prefix="/calculator", tags=["calculator"])
api_router.include_router(benchmark.router, prefix="/benchmark", tags=["benchmark"])
api_router.include_router(agent.router, prefix="/agent", tags=["agent"])
