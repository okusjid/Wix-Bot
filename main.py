from langchain_core.messages import HumanMessage

"""
    Starting Point of the Application
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.requests import Request
import warnings
from services.gpt import ChatbotSingleton


app = FastAPI(title="Wixchat")

app.add_middleware(
    CORSMiddleware,
    allow_origins= ["*"], #[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


warnings.filterwarnings("ignore", category=UserWarning, message="Unsupported Windows version")

chatbot = ChatbotSingleton()
chat_history = []

@app.get("/")
def chatbot_endpoint(request: Request):
    if chatbot.error:
        return {"answer": chatbot.error}

    ai_msg = None
    return {"answer": ai_msg.content if isinstance(ai_msg, str) else getattr(ai_msg, 'content', '')}

@app.post("/")
def chatbot_endpoint_post(request: Request):
    if chatbot.error:
        return {"answer": chatbot.error}
    
    rag_chain = chatbot.rag_chain
    ai_msg = None  # Define a default value for ai_msg

    data = request.json()
    question = data.get("question", "")
    ai_msg = rag_chain.invoke({"question": question, "chat_history": chat_history})
    chat_history.extend([HumanMessage(content=question), ai_msg])

    print("Updated chat_history:", chat_history)
    return {"answer": ai_msg.content if isinstance(ai_msg, str) else getattr(ai_msg, 'content', '')}