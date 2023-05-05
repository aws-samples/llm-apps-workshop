
from .endpoints import llm_ep
from fastapi import APIRouter

router = APIRouter()
router.include_router(llm_ep.router, prefix="/llm", tags=["llm"])
