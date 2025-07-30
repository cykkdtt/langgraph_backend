所有的示例代码都在`./examples`目录下
所有的测试代码都在`./tests`目录下
遇到不确定的问题时，查看`./spec/langgraph_offical_guide.md`和'https://langchain-ai.github.io/langgraph/cloud/reference/sdk/python_sdk_ref/'。
需要使用终端时，先激活虚拟环境后再执行命令。

## 阿里云嵌入模型配置规范

### 环境变量配置
- `DASHSCOPE_API_KEY`: 阿里云DashScope API密钥
- `LLM_EMBEDDING_MODEL`: 设置为 "openai:text-embedding-v4" (配置文件中的字符串标识)
- `LLM_EMBEDDING_DIMENSIONS`: 设置为 1024 (text-embedding-v4模型的向量维度)

### 代码中使用嵌入模型
1. **正确的导入方式**:
   ```python
   from langchain_community.embeddings import DashScopeEmbeddings
   ```

2. **正确的实例化方式**:
   ```python
   embeddings = DashScopeEmbeddings(
       model="text-embedding-v4",
       dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
   )
   ```

3. **在AsyncPostgresStore中使用**:
   ```python
   async with AsyncPostgresStore.from_conn_string(
       postgres_uri,
       index={
           "embed": embeddings,  # 使用DashScopeEmbeddings实例
           "dims": 1024,
           "fields": ["$"]  # 或指定具体字段如 ["content", "summary", "description"]
       }
   ) as store:
       # 使用store进行向量操作
   ```

### 重要注意事项
- **不要使用** `init_embeddings("openai:text-embedding-v4")`，这会导致OpenAI兼容接口参数错误
- **不要使用** OpenAI的嵌入模型类来调用阿里云API，会出现参数格式不兼容问题
- **必须使用** `DashScopeEmbeddings` 专门的类来确保API调用的正确性
- **确保** 在使用前通过 `load_dotenv()` 加载环境变量

### 向量存储配置
- 使用 `AsyncPostgresStore.from_conn_string()` 方法初始化
- 在 `async with` 语句中使用以确保连接正确管理
- 语义搜索使用位置参数: `store.asearch(namespace, query="搜索内容", limit=5)`