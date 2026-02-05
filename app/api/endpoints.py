from __future__ import annotations

import os
import tempfile
from pathlib import Path

from celery.result import AsyncResult
from fastapi import APIRouter, Depends, UploadFile

from app.api.deps import get_current_user
from app.core.agent import build_agent_graph
from app.core.retrieval import SecureRetriever
from app.ingestion.worker import celery_app, process_document
from app.models.schemas import ChatResponse, UserContext, UserRequest


router = APIRouter()
_workflow = None


def _get_workflow():
    global _workflow
    if _workflow is None:
        retriever = SecureRetriever()
        _workflow = build_agent_graph(retriever)
    return _workflow


@router.post("/upload")
async def upload_document(
    file: UploadFile,
    user: UserContext = Depends(get_current_user),
):
    suffix = Path(file.filename or "").suffix or ".bin"
    temp_dir = Path(os.getenv("UPLOAD_TEMP_DIR", tempfile.gettempdir()))
    temp_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=temp_dir) as tmp:
        contents = await file.read()
        tmp.write(contents)
        file_path = tmp.name

    task = process_document.delay(file_path)
    return {
        "filename": file.filename,
        "status": "queued",
        "task_id": task.id,
        "uploaded_by": user.user_id,
    }


@router.get("/tasks/{task_id}")
async def task_status(task_id: str):
    task = AsyncResult(task_id, app=celery_app)
    response = {"task_id": task_id, "status": task.status}
    if task.successful():
        response["result"] = task.result
    elif task.failed():
        response["error"] = str(task.result)
    return response


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: UserRequest,
    user: UserContext = Depends(get_current_user),
):
    workflow = _get_workflow()
    state = {
        "question": request.query,
        "generation": "",
        "documents": [],
        "user_roles": [request.role],
        "steps": [],
        "rewrite_query": False,
        "retries": 0,
        "route": "",
        "grounded": False,
        "hallucination_retries": 0,
        "force_grounded": False,
    }
    result = workflow.invoke(state)
    documents = result.get("documents", [])
    citations = [
        doc.metadata.get("source")
        for doc in documents
        if doc.metadata.get("source")
    ]
    return ChatResponse(
        answer=result.get("generation", ""),
        citations=citations,
        user_id=request.user_id,
        steps=result.get("steps", []),
    )
