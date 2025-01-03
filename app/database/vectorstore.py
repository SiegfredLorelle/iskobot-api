from langchain_community.vectorstores.pgvector import PGVector
from .connector import get_db_connection
from langchain_google_vertexai import VertexAIEmbeddings


def initialize_vectorstore(delete_on_insert=False):
    return PGVector(
        connection_string="postgresql+pg8000://",
        use_jsonb=True,
        engine_args=dict(
            creator=get_db_connection,
        ),
        embedding_function=VertexAIEmbeddings(
            model_name="text-embedding-005"
        ),
    pre_delete_collection=delete_on_insert
    )