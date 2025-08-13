"""
RAG系统相关数据模型

定义RAG（检索增强生成）系统API的请求和响应模型。
"""

from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class DocumentType(str, Enum):
    """文档类型"""
    TEXT = "text"
    PDF = "pdf"
    WORD = "word"
    MARKDOWN = "markdown"
    HTML = "html"
    JSON = "json"
    CSV = "csv"
    XML = "xml"


class DocumentStatus(str, Enum):
    """文档状态"""
    PENDING = "pending"
    PROCESSING = "processing"
    INDEXED = "indexed"
    FAILED = "failed"
    ARCHIVED = "archived"


class ChunkingStrategy(str, Enum):
    """分块策略"""
    FIXED_SIZE = "fixed_size"
    SEMANTIC = "semantic"
    SENTENCE = "sentence"
    PARAGRAPH = "paragraph"
    RECURSIVE = "recursive"


class EmbeddingModel(str, Enum):
    """嵌入模型"""
    OPENAI_ADA = "openai_ada"
    OPENAI_3_SMALL = "openai_3_small"
    OPENAI_3_LARGE = "openai_3_large"
    SENTENCE_TRANSFORMERS = "sentence_transformers"
    BGE_LARGE = "bge_large"
    BGE_BASE = "bge_base"


class VectorStore(str, Enum):
    """向量存储"""
    CHROMA = "chroma"
    PINECONE = "pinecone"
    WEAVIATE = "weaviate"
    QDRANT = "qdrant"
    FAISS = "faiss"


class DocumentModel(BaseModel):
    """文档模型"""
    id: str = Field(description="文档ID")
    title: str = Field(description="文档标题")
    content: str = Field(description="文档内容")
    type: DocumentType = Field(description="文档类型")
    status: DocumentStatus = Field(description="文档状态")
    source: Optional[str] = Field(None, description="文档来源")
    url: Optional[str] = Field(None, description="文档URL")
    file_path: Optional[str] = Field(None, description="文件路径")
    file_size: Optional[int] = Field(None, description="文件大小")
    created_at: datetime = Field(description="创建时间")
    updated_at: datetime = Field(description="更新时间")
    indexed_at: Optional[datetime] = Field(None, description="索引时间")
    chunk_count: int = Field(0, description="分块数量")
    tags: List[str] = Field(default_factory=list, description="标签")
    metadata: Optional[Dict[str, Any]] = Field(None, description="元数据")


class DocumentChunk(BaseModel):
    """文档分块"""
    id: str = Field(description="分块ID")
    document_id: str = Field(description="文档ID")
    content: str = Field(description="分块内容")
    chunk_index: int = Field(description="分块索引")
    start_char: int = Field(description="起始字符位置")
    end_char: int = Field(description="结束字符位置")
    embedding: Optional[List[float]] = Field(None, description="向量嵌入")
    metadata: Optional[Dict[str, Any]] = Field(None, description="元数据")


class DocumentUploadRequest(BaseModel):
    """文档上传请求"""
    title: str = Field(description="文档标题")
    content: Optional[str] = Field(None, description="文档内容")
    type: DocumentType = Field(description="文档类型")
    source: Optional[str] = Field(None, description="文档来源")
    url: Optional[str] = Field(None, description="文档URL")
    tags: List[str] = Field(default_factory=list, description="标签")
    metadata: Optional[Dict[str, Any]] = Field(None, description="元数据")
    chunking_config: Optional[Dict[str, Any]] = Field(None, description="分块配置")
    embedding_config: Optional[Dict[str, Any]] = Field(None, description="嵌入配置")


class DocumentUpdateRequest(BaseModel):
    """文档更新请求"""
    title: Optional[str] = Field(None, description="文档标题")
    content: Optional[str] = Field(None, description="文档内容")
    status: Optional[DocumentStatus] = Field(None, description="文档状态")
    tags: Optional[List[str]] = Field(None, description="标签")
    metadata: Optional[Dict[str, Any]] = Field(None, description="元数据")


class DocumentSearchRequest(BaseModel):
    """文档搜索请求"""
    query: str = Field(description="搜索查询")
    document_types: Optional[List[DocumentType]] = Field(None, description="文档类型过滤")
    tags: Optional[List[str]] = Field(None, description="标签过滤")
    limit: int = Field(10, ge=1, le=100, description="返回数量限制")
    similarity_threshold: float = Field(0.7, ge=0.0, le=1.0, description="相似度阈值")
    include_content: bool = Field(True, description="是否包含内容")


class DocumentSearchResponse(BaseModel):
    """文档搜索响应"""
    documents: List[DocumentModel] = Field(description="搜索结果")
    total: int = Field(description="总数量")
    query: str = Field(description="搜索查询")
    search_time: float = Field(description="搜索耗时")


class VectorSearchRequest(BaseModel):
    """向量搜索请求"""
    query: str = Field(description="搜索查询")
    collection_name: Optional[str] = Field(None, description="集合名称")
    top_k: int = Field(10, ge=1, le=100, description="返回数量")
    similarity_threshold: float = Field(0.7, ge=0.0, le=1.0, description="相似度阈值")
    filters: Optional[Dict[str, Any]] = Field(None, description="过滤条件")
    include_metadata: bool = Field(True, description="是否包含元数据")


class VectorSearchResult(BaseModel):
    """向量搜索结果"""
    chunk_id: str = Field(description="分块ID")
    document_id: str = Field(description="文档ID")
    content: str = Field(description="内容")
    score: float = Field(description="相似度分数")
    metadata: Optional[Dict[str, Any]] = Field(None, description="元数据")


class VectorSearchResponse(BaseModel):
    """向量搜索响应"""
    results: List[VectorSearchResult] = Field(description="搜索结果")
    total: int = Field(description="总数量")
    query: str = Field(description="搜索查询")
    search_time: float = Field(description="搜索耗时")


class RetrievalConfig(BaseModel):
    """检索配置"""
    embedding_model: EmbeddingModel = Field(description="嵌入模型")
    vector_store: VectorStore = Field(description="向量存储")
    chunking_strategy: ChunkingStrategy = Field(description="分块策略")
    chunk_size: int = Field(1000, ge=100, le=10000, description="分块大小")
    chunk_overlap: int = Field(200, ge=0, le=1000, description="分块重叠")
    top_k: int = Field(5, ge=1, le=50, description="检索数量")
    similarity_threshold: float = Field(0.7, ge=0.0, le=1.0, description="相似度阈值")
    rerank: bool = Field(False, description="是否重排序")
    rerank_model: Optional[str] = Field(None, description="重排序模型")


class RAGRequest(BaseModel):
    """RAG请求"""
    query: str = Field(description="查询问题")
    collection_name: Optional[str] = Field(None, description="集合名称")
    retrieval_config: Optional[RetrievalConfig] = Field(None, description="检索配置")
    generation_config: Optional[Dict[str, Any]] = Field(None, description="生成配置")
    context_window: int = Field(4000, ge=1000, le=32000, description="上下文窗口")
    include_sources: bool = Field(True, description="是否包含来源")


class RAGResponse(BaseModel):
    """RAG响应"""
    answer: str = Field(description="生成的答案")
    sources: List[VectorSearchResult] = Field(description="检索的来源")
    query: str = Field(description="原始查询")
    retrieval_time: float = Field(description="检索耗时")
    generation_time: float = Field(description="生成耗时")
    total_time: float = Field(description="总耗时")
    metadata: Optional[Dict[str, Any]] = Field(None, description="元数据")


class IndexingRequest(BaseModel):
    """索引请求"""
    document_ids: List[str] = Field(description="文档ID列表")
    force_reindex: bool = Field(False, description="是否强制重新索引")
    chunking_config: Optional[Dict[str, Any]] = Field(None, description="分块配置")
    embedding_config: Optional[Dict[str, Any]] = Field(None, description="嵌入配置")


class IndexingResponse(BaseModel):
    """索引响应"""
    task_id: str = Field(description="任务ID")
    status: str = Field(description="索引状态")
    document_count: int = Field(description="文档数量")
    estimated_duration: Optional[int] = Field(None, description="预估耗时（秒）")


class CollectionInfo(BaseModel):
    """集合信息"""
    name: str = Field(description="集合名称")
    description: Optional[str] = Field(None, description="集合描述")
    document_count: int = Field(description="文档数量")
    chunk_count: int = Field(description="分块数量")
    embedding_model: str = Field(description="嵌入模型")
    vector_store: str = Field(description="向量存储")
    created_at: datetime = Field(description="创建时间")
    updated_at: datetime = Field(description="更新时间")
    metadata: Optional[Dict[str, Any]] = Field(None, description="元数据")


class RAGStats(BaseModel):
    """RAG统计"""
    total_documents: int = Field(description="总文档数")
    total_chunks: int = Field(description="总分块数")
    total_queries: int = Field(description="总查询数")
    avg_retrieval_time: float = Field(description="平均检索时间")
    avg_generation_time: float = Field(description="平均生成时间")
    success_rate: float = Field(description="成功率")
    last_indexed: Optional[datetime] = Field(None, description="最后索引时间")
    storage_size: int = Field(description="存储大小（字节）")