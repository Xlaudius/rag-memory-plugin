"""
Tool handlers and schemas for RAG Memory Plugin
"""

from .schemas import (
    RAG_SEARCH,
    RAG_ADD_DOCUMENT,
    RAG_GET_PEER_CONTEXT,
    RAG_GET_SESSION_CONTEXT,
    RAG_START_SESSION,
    RAG_END_SESSION,
    RAG_CAPTURE_MESSAGE,
    RAG_LIST_PEERS,
    RAG_LIST_SESSIONS
)
from .handlers import (
    rag_search,
    rag_add_document,
    rag_get_peer_context,
    rag_get_session_context,
    rag_start_session,
    rag_end_session,
    rag_capture_message,
    rag_list_peers,
    rag_list_sessions,
    inject_context,
    capture_output,
    set_managers
)

__all__ = [
    # Schemas
    'RAG_SEARCH', 'RAG_ADD_DOCUMENT', 'RAG_GET_PEER_CONTEXT',
    'RAG_GET_SESSION_CONTEXT', 'RAG_START_SESSION', 'RAG_END_SESSION',
    'RAG_CAPTURE_MESSAGE', 'RAG_LIST_PEERS', 'RAG_LIST_SESSIONS',
    # Handlers
    'rag_search', 'rag_add_document', 'rag_get_peer_context',
    'rag_get_session_context', 'rag_start_session', 'rag_end_session',
    'rag_capture_message', 'rag_list_peers', 'rag_list_sessions',
    'inject_context', 'capture_output', 'set_managers'
]
