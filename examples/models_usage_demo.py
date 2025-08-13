#!/usr/bin/env python3
"""
Models目录使用演示

本脚本演示如何使用 /models 目录中定义的各种数据模型，包括：
1. 基础响应模型的使用
2. 聊天相关模型的使用
3. 智能体模型的使用
4. 记忆管理模型的使用
5. RAG系统模型的使用
6. 数据验证和序列化
7. API接口中的实际应用
"""

import sys
import os
import json
from datetime import datetime
from typing import Dict, Any, List

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入各种数据模型
from models.base_models import (
    BaseResponse, SuccessResponse, ErrorResponse,
    PaginatedResponse, HealthCheckResponse, ValidationError
)
from models.chat_models import (
    ChatRequest, ChatResponse, Message, MessageRole, MessageType,
    StreamChunk, ThreadInfo, ToolCall, ToolResult
)
from models.agent_models import (
    AgentInfo, CreateAgentRequest, AgentInstanceRequest, AgentInstanceResponse,
    AgentType, AgentStatus, AgentConfig, AgentTool, AgentCapability
)
from models.memory_models import (
    MemoryItem, MemoryCreateRequest, MemorySearchRequest, MemorySearchResponse,
    MemoryType, MemoryImportance, MemoryStatus
)
from models.rag_models import (
    DocumentModel, DocumentUploadRequest, VectorSearchRequest, VectorSearchResponse,
    RAGRequest, RAGResponse, DocumentType, DocumentStatus
)


def demo_base_models():
    """演示基础响应模型的使用"""
    print("=" * 60)
    print("1. 基础响应模型演示")
    print("=" * 60)
    
    # 成功响应示例
    success_response = SuccessResponse(
        message="操作成功",
        data={"result": "数据处理完成", "count": 100},
        request_id="req_123456"
    )
    print("✅ 成功响应:")
    print(json.dumps(success_response.dict(), indent=2, ensure_ascii=False, default=str))
    
    # 错误响应示例
    error_response = ErrorResponse(
        message="操作失败",
        error="参数验证错误",
        request_id="req_123457"
    )
    print("\n❌ 错误响应:")
    print(json.dumps(error_response.dict(), indent=2, ensure_ascii=False, default=str))
    
    # 分页响应示例
    from models.base_models import PaginationInfo
    pagination = PaginationInfo(
        page=1,
        page_size=20,
        total=100,
        total_pages=5,
        has_next=True,
        has_prev=False
    )
    
    paginated_response = PaginatedResponse[Dict[str, Any]](
        success=True,
        message="获取数据成功",
        items=[{"id": i, "name": f"项目{i}"} for i in range(1, 21)],
        pagination=pagination
    )
    print("\n📄 分页响应:")
    print(json.dumps(paginated_response.dict(), indent=2, ensure_ascii=False, default=str))


def demo_chat_models():
    """演示聊天相关模型的使用"""
    print("\n" + "=" * 60)
    print("2. 聊天模型演示")
    print("=" * 60)
    
    # 聊天请求示例
    chat_request = ChatRequest(
        message="你好，请帮我分析一下市场趋势",
        agent_id="agent_001",
        stream=True,
        temperature=0.7,
        max_tokens=1000,
        tools=["web_search", "data_analysis"],
        context={"user_preference": "详细分析", "language": "zh-CN"}
    )
    print("💬 聊天请求:")
    print(json.dumps(chat_request.dict(), indent=2, ensure_ascii=False, default=str))
    
    # 消息模型示例
    message = Message(
        id="msg_001",
        role=MessageRole.ASSISTANT,
        content="根据最新的市场数据分析，当前市场呈现以下趋势...",
        timestamp=datetime.now(),
        metadata={"confidence": 0.95, "sources": ["market_data", "news_analysis"]}
    )
    print("\n📝 消息模型:")
    print(json.dumps(message.dict(), indent=2, ensure_ascii=False, default=str))
    
    # 工具调用示例
    tool_call = ToolCall(
        id="tool_call_001",
        name="web_search",
        arguments={"query": "2024年市场趋势分析", "limit": 10}
    )
    
    tool_result = ToolResult(
        tool_call_id="tool_call_001",
        result={"results": ["结果1", "结果2"], "total": 2}
    )
    print("\n🔧 工具调用:")
    print(json.dumps(tool_call.dict(), indent=2, ensure_ascii=False, default=str))
    print("\n🔧 工具结果:")
    print(json.dumps(tool_result.dict(), indent=2, ensure_ascii=False, default=str))


def demo_agent_models():
    """演示智能体模型的使用"""
    print("\n" + "=" * 60)
    print("3. 智能体模型演示")
    print("=" * 60)
    
    # 智能体工具配置
    agent_tools = [
        AgentTool(
            name="web_search",
            description="网络搜索工具",
            enabled=True,
            config={"api_key": "***", "max_results": 10}
        ),
        AgentTool(
            name="data_analysis",
            description="数据分析工具",
            enabled=True,
            config={"precision": "high"}
        )
    ]
    
    # 智能体能力配置
    agent_capabilities = [
        AgentCapability(
            name="natural_language_processing",
            description="自然语言处理能力",
            enabled=True,
            config={"model": "gpt-4", "language": "zh-CN"}
        ),
        AgentCapability(
            name="data_visualization",
            description="数据可视化能力",
            enabled=True
        )
    ]
    
    # 智能体配置
    agent_config = AgentConfig(
        model="gpt-4-turbo",
        temperature=0.7,
        max_tokens=2000,
        system_prompt="你是一个专业的市场分析师，擅长数据分析和趋势预测。",
        tools=agent_tools,
        capabilities=agent_capabilities,
        memory_config={"type": "long_term", "capacity": 10000},
        rag_config={"enabled": True, "collection": "market_data"}
    )
    
    # 创建智能体请求
    create_agent_request = CreateAgentRequest(
        name="市场分析师",
        description="专业的市场趋势分析智能体",
        type=AgentType.RAG,
        config=agent_config,
        metadata={"department": "finance", "version": "1.0"}
    )
    print("🤖 创建智能体请求:")
    print(json.dumps(create_agent_request.dict(), indent=2, ensure_ascii=False, default=str))
    
    # 智能体实例响应
    agent_instance_response = AgentInstanceResponse(
        instance_id="inst_001",
        agent_id="agent_001",
        status="active",
        created_at=datetime.now(),
        config={"runtime_config": "optimized"},
        metadata={"performance": "high"}
    )
    print("\n🤖 智能体实例响应:")
    print(json.dumps(agent_instance_response.dict(), indent=2, ensure_ascii=False, default=str))


def demo_memory_models():
    """演示记忆管理模型的使用"""
    print("\n" + "=" * 60)
    print("4. 记忆管理模型演示")
    print("=" * 60)
    
    # 创建记忆请求
    memory_create_request = MemoryCreateRequest(
        content="用户偏好使用详细的数据分析报告，关注长期趋势",
        type=MemoryType.LONG_TERM,
        importance=MemoryImportance.HIGH,
        agent_id="agent_001",
        thread_id="thread_001",
        tags=["用户偏好", "分析风格", "长期记忆"],
        metadata={"category": "user_preference", "confidence": 0.9}
    )
    print("🧠 创建记忆请求:")
    print(json.dumps(memory_create_request.dict(), indent=2, ensure_ascii=False, default=str))
    
    # 记忆项示例
    memory_item = MemoryItem(
        id="mem_001",
        content="用户在2024年1月询问过市场趋势，对技术分析很感兴趣",
        type=MemoryType.EPISODIC,
        importance=MemoryImportance.MEDIUM,
        status=MemoryStatus.ACTIVE,
        agent_id="agent_001",
        thread_id="thread_001",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        access_count=5,
        tags=["市场趋势", "技术分析", "用户兴趣"],
        metadata={"session": "2024-01", "topic": "market_analysis"},
        related_memories=["mem_002", "mem_003"]
    )
    print("\n🧠 记忆项:")
    print(json.dumps(memory_item.dict(), indent=2, ensure_ascii=False, default=str))
    
    # 记忆搜索请求
    memory_search_request = MemorySearchRequest(
        query="用户对市场分析的偏好",
        agent_id="agent_001",
        memory_types=[MemoryType.LONG_TERM, MemoryType.EPISODIC],
        importance_levels=[MemoryImportance.HIGH, MemoryImportance.MEDIUM],
        tags=["用户偏好", "市场分析"],
        limit=10,
        similarity_threshold=0.8
    )
    print("\n🔍 记忆搜索请求:")
    print(json.dumps(memory_search_request.dict(), indent=2, ensure_ascii=False, default=str))


def demo_rag_models():
    """演示RAG系统模型的使用"""
    print("\n" + "=" * 60)
    print("5. RAG系统模型演示")
    print("=" * 60)
    
    # 文档上传请求
    document_upload_request = DocumentUploadRequest(
        title="2024年市场趋势报告",
        content="本报告分析了2024年全球市场的主要趋势...",
        type=DocumentType.PDF,
        source="market_research_team",
        tags=["市场趋势", "2024", "分析报告"],
        metadata={"author": "分析师团队", "department": "research"},
        chunking_config={"strategy": "semantic", "chunk_size": 1000},
        embedding_config={"model": "openai_3_large"}
    )
    print("📄 文档上传请求:")
    print(json.dumps(document_upload_request.dict(), indent=2, ensure_ascii=False, default=str))
    
    # 文档模型示例
    document_model = DocumentModel(
        id="doc_001",
        title="2024年市场趋势报告",
        content="详细的市场分析内容...",
        type=DocumentType.PDF,
        status=DocumentStatus.INDEXED,
        source="market_research_team",
        file_size=2048000,
        created_at=datetime.now(),
        updated_at=datetime.now(),
        indexed_at=datetime.now(),
        chunk_count=50,
        tags=["市场趋势", "2024", "分析报告"],
        metadata={"version": "1.0", "language": "zh-CN"}
    )
    print("\n📄 文档模型:")
    print(json.dumps(document_model.dict(), indent=2, ensure_ascii=False, default=str))
    
    # RAG请求示例
    from models.rag_models import RetrievalConfig, EmbeddingModel, VectorStore, ChunkingStrategy
    
    retrieval_config = RetrievalConfig(
        embedding_model=EmbeddingModel.OPENAI_3_LARGE,
        vector_store=VectorStore.CHROMA,
        chunking_strategy=ChunkingStrategy.SEMANTIC,
        chunk_size=1000,
        chunk_overlap=200,
        top_k=5,
        similarity_threshold=0.8,
        rerank=True,
        rerank_model="bge-reranker"
    )
    
    rag_request = RAGRequest(
        query="2024年市场的主要趋势是什么？",
        collection_name="market_reports",
        retrieval_config=retrieval_config,
        generation_config={"temperature": 0.3, "max_tokens": 1000},
        context_window=4000,
        include_sources=True
    )
    print("\n🔍 RAG请求:")
    print(json.dumps(rag_request.dict(), indent=2, ensure_ascii=False, default=str))


def demo_data_validation():
    """演示数据验证功能"""
    print("\n" + "=" * 60)
    print("6. 数据验证演示")
    print("=" * 60)
    
    # 正确的数据验证
    try:
        valid_chat_request = ChatRequest(
            message="测试消息",
            agent_id="agent_001",
            temperature=0.5,  # 有效范围内
            max_tokens=1000   # 有效值
        )
        print("✅ 数据验证成功:")
        print(f"   消息: {valid_chat_request.message}")
        print(f"   温度: {valid_chat_request.temperature}")
        print(f"   最大tokens: {valid_chat_request.max_tokens}")
    except Exception as e:
        print(f"❌ 验证失败: {e}")
    
    # 错误的数据验证
    try:
        invalid_chat_request = ChatRequest(
            message="测试消息",
            agent_id="agent_001",
            temperature=3.0,  # 超出有效范围 (0.0-2.0)
            max_tokens=0      # 无效值 (必须 >= 1)
        )
    except Exception as e:
        print(f"\n❌ 预期的验证错误: {e}")
    
    # 分页参数验证
    try:
        from models.memory_models import MemoryListRequest
        valid_list_request = MemoryListRequest(
            agent_id="agent_001",
            page=1,
            page_size=20
        )
        print(f"\n✅ 分页验证成功: 页码={valid_list_request.page}, 页大小={valid_list_request.page_size}")
    except Exception as e:
        print(f"❌ 分页验证失败: {e}")


def demo_api_usage():
    """演示在API中的实际使用"""
    print("\n" + "=" * 60)
    print("7. API使用演示")
    print("=" * 60)
    
    print("📋 在FastAPI中的使用示例:")
    print("""
# 在API路由中使用数据模型

from fastapi import APIRouter, HTTPException
from models.chat_models import ChatRequest, ChatResponse
from models.base_models import BaseResponse

router = APIRouter()

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    '''聊天API端点'''
    try:
        # 自动验证请求数据
        # request.message, request.agent_id 等字段已验证
        
        # 处理聊天逻辑
        response_content = await process_chat(request)
        
        # 返回标准化响应
        return ChatResponse(
            message=response_content,
            thread_id=request.thread_id or "default",
            agent_id=request.agent_id
        )
    except Exception as e:
        # 返回错误响应
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health", response_model=BaseResponse)
async def health_endpoint():
    '''健康检查端点'''
    return BaseResponse(
        success=True,
        message="系统运行正常",
        data={"status": "healthy", "timestamp": datetime.now()}
    )
    """)
    
    print("\n🔄 数据序列化和反序列化:")
    
    # JSON序列化
    chat_request = ChatRequest(
        message="测试消息",
        agent_id="agent_001"
    )
    json_data = chat_request.json()
    print(f"序列化为JSON: {json_data}")
    
    # JSON反序列化
    parsed_request = ChatRequest.parse_raw(json_data)
    print(f"从JSON解析: {parsed_request.message}")
    
    # 字典转换
    dict_data = chat_request.dict()
    print(f"转换为字典: {dict_data}")
    
    # 从字典创建
    new_request = ChatRequest(**dict_data)
    print(f"从字典创建: {new_request.agent_id}")


def main():
    """主函数"""
    print("🚀 Models目录使用演示")
    print("本演示展示了 /models 目录中各种数据模型的使用方法")
    
    # 运行各个演示
    demo_base_models()
    demo_chat_models()
    demo_agent_models()
    demo_memory_models()
    demo_rag_models()
    demo_data_validation()
    demo_api_usage()
    
    print("\n" + "=" * 60)
    print("✨ 演示完成！")
    print("=" * 60)
    print("""
📚 Models目录使用总结:

1. 📁 目录结构:
   - base_models.py    # 基础响应模型
   - chat_models.py    # 聊天相关模型
   - agent_models.py   # 智能体模型
   - memory_models.py  # 记忆管理模型
   - rag_models.py     # RAG系统模型

2. 🎯 主要功能:
   - 数据验证和类型检查
   - 自动序列化/反序列化
   - API请求/响应标准化
   - 文档生成支持

3. 💡 使用建议:
   - 在API端点中使用作为请求/响应模型
   - 利用Pydantic的验证功能确保数据质量
   - 使用枚举类型提高代码可读性
   - 合理设置字段约束和默认值

4. 🔧 扩展方式:
   - 继承基础模型类
   - 添加自定义验证器
   - 使用Field()设置字段属性
   - 定义模型间的关系
    """)


if __name__ == "__main__":
    main()