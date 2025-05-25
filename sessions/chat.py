from fastapi import APIRouter, Depends, HTTPException, Security
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import uuid
from typing import Optional
from .models import QueryWithSession, ChatResponse, MessageRole
from .service import SessionService
from .auth import AuthService
from .memory import SessionMemory
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

def create_chat_router(supabase_client, llm, knowledge_bank_retriever, retry_with_backoff):
    """Factory function to create the chat router with context awareness"""
    router = APIRouter(prefix="/chat", tags=["chat"])
    session_service = SessionService(supabase_client)
    auth_service = AuthService(supabase_client)
    security = HTTPBearer(auto_error=False)
    
    async def get_optional_user(
        credentials: HTTPAuthorizationCredentials = Security(security)
    ):
        if credentials:
            try:
                return await auth_service.get_current_user(credentials.credentials)
            except HTTPException:
                pass
        return None

    context_aware_prompt = PromptTemplate.from_template("""\
    You're **Iskobot**, a helpful and knowledgeable assistant in Computer Engineering.

    *When responding:**
    - Answer only using the provided knowledge bank. If the topic isn’t covered, say something like:
    *"That's a bit outside what I know right now. My focus is on Computer Engineering, but I'm happy to help with that if it relates!"*
    - Keep answers **clear, concise, and accurate**.
    - Use **bullet points** where helpful.
    - Match the language and tone of the user's question.
    - Don’t mention the knowledge bank or where the info came from.
    - Focus on **technical accuracy** for anything related to Computer Engineering.
    - When it fits naturally, end with a helpful follow-up like:
    *"Would you also like to know more about [related topic]?"*

    **Knowledge Bank:**
    {knowledge_bank}

    **Conversation History:**
    {context_section}

    **New Question:**
    {query}

    **Your Answer:**""")
    @router.post("/", response_model=ChatResponse)
    async def chat_with_context(
        request: QueryWithSession,
        user_id: Optional[uuid.UUID] = Depends(get_optional_user)
    ):
        """Enhanced chat endpoint supporting both authenticated and anonymous sessions"""
        try:
            session_id = await validate_or_create_session(request, user_id)
            await validate_session_access(session_id, user_id)

            # Initialize memory with proper context
            memory = SessionMemory(session_service, session_id, user_id)
            await memory.load_history()

            # Process message and generate response
            return await process_chat_interaction(
                request.query, 
                session_id, 
                memory, 
                user_id
            )

        except HTTPException as he:
            raise he
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    async def validate_or_create_session(request: QueryWithSession, user_id: Optional[uuid.UUID]):
        """Handle session creation/validation logic"""
        if request.session_id:
            return request.session_id
        
        title = generate_session_title(request.query)
        session = await session_service.create_session(user_id, title)
        return session.id

    async def validate_session_access(session_id: uuid.UUID, user_id: Optional[uuid.UUID]):
        """Verify session ownership or anonymous access"""
        session = await session_service.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Validate session ownership for authenticated users
        if user_id and session.user_id != user_id:
            raise HTTPException(status_code=403, detail="Session access denied")
        
        # Prevent authenticated access to anonymous sessions
        if not user_id and session.user_id:
            raise HTTPException(status_code=403, detail="Invalid session type")

    async def process_chat_interaction(query: str, session_id: uuid.UUID, memory: SessionMemory, user_id: Optional[uuid.UUID]):
        """Handle message processing and response generation"""
        # Store user message
        user_message = await session_service.add_message(session_id, MessageRole.USER, query)
        
        # Generate context-aware response
        answer = await generate_ai_response(query, memory)
        
        # Store assistant message
        assistant_message = await session_service.add_message(
            session_id, 
            MessageRole.ASSISTANT, 
            answer
        )

        return ChatResponse(
            response=answer,
            session_id=session_id,
            message_id=assistant_message.id
        )

    async def generate_ai_response(query: str, memory: SessionMemory):
        """Generate context-aware AI response"""
        context_chain = (
            RunnableParallel({
                "knowledge_bank": knowledge_bank_retriever,
                "query": RunnablePassthrough(),
                "context_section": lambda x: format_context_section(memory)
            })
            | context_aware_prompt
            | llm
            | StrOutputParser()
        )

        async def invoke_chain():
            return await context_chain.ainvoke(query)

        return await retry_with_backoff(invoke_chain)

    def format_context_section(memory: SessionMemory):
        """Format conversation context for prompt"""
        conversation_context = memory.get_context()
        return f"**Previous conversation:**\n{conversation_context}\n" if conversation_context else ""

    def generate_session_title(query: str):
        """Generate automatic session title from query"""
        return query[:50] + "..." if len(query) > 50 else query

    return router