from fastapi import APIRouter

router = APIRouter()

@router.get("/health")
async def get_user():
    return {
        "user_id": "user_id",
        "name": "Harshit",
    }
