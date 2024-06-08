from typing import List
from fastapi import status, HTTPException, APIRouter, Body
from app.agent_service import conversational_agent
from app import service_schemas 


router = APIRouter(
    prefix="/agentservice",
    tags=['Agentservice']
)


@router.post("/", response_model=List[service_schemas.ChatMessage])  
async def chat(user_input: str = Body(...)):
    if not conversational_agent:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                      detail="Access to the model failed.")
    response = conversational_agent.run(user_input)
    # conversational_agent.get_full_history_generator()
    return response #response[-1] #use [-1] if you want only the last one and response_model=schemas.ChatMessage

