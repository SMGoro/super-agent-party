# mcp_client_fixed.py
import json
import asyncio
import logging
import shutil
from typing import Dict, Any, AsyncIterator, Optional, List
from contextlib import AsyncExitStack, asynccontextmanager

# 全局缓存命令路径，避免每次连接都 IO 查找
_CMD_PATH_CACHE = {}

def get_command_path(command_name: str, default_command: str = "uv") -> str:
    cache_key = (command_name, default_command)
    if cache_key in _CMD_PATH_CACHE:
        return _CMD_PATH_CACHE[cache_key]
    
    path = shutil.which(command_name) or shutil.which(default_command)
    if not path:
        raise FileNotFoundError(f"未找到 {command_name} 或 {default_command}")
    
    _CMD_PATH_CACHE[cache_key] = path
    return path

# ---------- 连接管理 ----------
class ConnectionManager:
    def __init__(self) -> None:
        self.session = None

    @asynccontextmanager
    async def connect(self, config: dict) -> AsyncIterator["ConnectionManager"]:
        # 【优化1】懒加载 MCP 库，防止拖慢主程序启动速度
        # 只有在真正需要建立连接时才加载这些库
        import anyio
        from mcp import ClientSession
        from mcp.client.stdio import stdio_client, StdioServerParameters
        from mcp.client.sse import sse_client
        from mcp.client.websocket import websocket_client
        from mcp.client.streamable_http import streamablehttp_client

        async with AsyncExitStack() as stack:
            # 1. 建立传输层
            if "command" in config:
                server_params = StdioServerParameters(
                    command=get_command_path(config["command"]),
                    args=config.get("args", []),
                    env=config.get("env"),
                )
                read, write = await stack.enter_async_context(stdio_client(server_params))
            else:
                mcptype = config.get("type", "ws")
                # 简化逻辑：标准化类型判断
                if "streamable" in mcptype:
                    mcptype = "streamablehttp"
                
                url = config["url"]
                headers = config.get("headers", {})
                
                if mcptype == "ws":
                    transport = await stack.enter_async_context(websocket_client(url, headers=headers))
                    read, write = transport
                elif mcptype == "sse":
                    transport = await stack.enter_async_context(sse_client(url, headers=headers))
                    read, write = transport
                    
                    # SSE 握手检查
                    try:
                        with anyio.move_on_after(3):
                            await read.receive()
                    except anyio.EndOfStream:
                        raise RuntimeError("SSE stream closed immediately")
                    except Exception as e:
                        raise RuntimeError(f"SSE initial handshake failed: {e}") from e
                elif mcptype == "streamablehttp":
                    transport = await stack.enter_async_context(streamablehttp_client(url, headers=headers))
                    read, write, _ = transport
                else:
                    raise ValueError(f"Unknown MCP type: {mcptype}")

            # 2. 建立会话
            self.session = await stack.enter_async_context(ClientSession(read, write))
            await self.session.initialize()
            
            # 注意：此处不再 list_tools，改为由 Client 统一管理缓存
            logging.info(f"Connected to MCP server: {config.get('url') or config.get('command')}")
            yield self


# ---------- 客户端 ----------
class McpClient:
    def __init__(self) -> None:
        self._conn: Optional[ConnectionManager] = None
        self._config: Optional[dict] = None
        self._lock = asyncio.Lock()
        self._monitor_task: Optional[asyncio.Task] = None
        self._shutdown = False
        self._on_failure_callback: Optional[callable] = None
        
        # 【优化2】增加本地缓存，避免每次调用 tool 都走网络请求
        self._tools_cache: List[Dict] = []  
        self._tools_openai_cache: List[Dict] = [] 
        self._disabled = False # 增加标记位

    @property
    def disabled(self):
        return self._disabled

    @disabled.setter
    def disabled(self, value):
        self._disabled = value

    async def initialize(self, server_name: str, server_config: dict, on_failure_callback: Optional[callable] = None) -> None:
        self._config = server_config
        self._on_failure_callback = on_failure_callback
        if self._monitor_task is None or self._monitor_task.done():
            self._monitor_task = asyncio.create_task(self._connection_monitor())

    async def close(self) -> None:
        self._shutdown = True
        if self._monitor_task and not self._monitor_task.done():
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass

    async def _refresh_tools_cache(self):
        """【优化3】独立的工具刷新逻辑"""
        if not self._conn or not self._conn.session:
            return
        try:
            # 仅在连接建立或重连时调用一次
            result = await self._conn.session.list_tools()
            tools = result.tools
            
            # 缓存原始数据
            self._tools_cache = tools
            
            # 预处理 OpenAI 格式（无需在每次 get_openai_functions 时转换）
            self._tools_openai_cache = []
            for t in tools:
                self._tools_openai_cache.append({
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.inputSchema,
                    },
                    # 额外存储一些元数据方便过滤
                    "_original_name": t.name 
                })
            logging.info(f"Tools cache refreshed. Count: {len(tools)}")
        except Exception as e:
            logging.error(f"Failed to refresh tools cache: {e}")

    async def _connection_monitor(self) -> None:
        while not self._shutdown and not self._disabled:
            try:
                async with ConnectionManager().connect(self._config) as conn:
                    async with self._lock:
                        self._conn = conn
                    
                    # 连接成功后，立即刷新一次缓存
                    await self._refresh_tools_cache()

                    # 心跳保持
                    while not self._shutdown:
                        try:
                            # 3秒超时，30秒间隔
                            await asyncio.wait_for(self._conn.session.send_ping(), timeout=3)
                        except Exception:
                            logging.warning("MCP Ping failed, reconnecting...")
                            break 
                        await asyncio.sleep(30)
            except Exception as e:
                # 避免刷屏日志
                if not self._shutdown and not self._disabled:
                    logging.error(f"MCP Connection error ({self._config.get('command') or self._config.get('url')}): {e}")
                    if self._on_failure_callback:
                        await self._on_failure_callback(str(e))
            finally:
                async with self._lock:
                    self._conn = None
                    # 连接断开，清空缓存，防止调用错误的 Session
                    self._tools_cache = []
                    self._tools_openai_cache = []
            
            if not self._shutdown:
                await asyncio.sleep(5)

    # ---------- 外部 API (高性能版) ----------
    async def get_openai_functions(self, disable_tools: List[str] = []) -> List[Dict]:
        """
        此处不再发起网络请求，直接读取内存缓存。
        速度从 ms/s 级提升到 μs 级。
        """
        if not self._tools_openai_cache:
            return []
        
        # 如果没有禁用列表，直接返回全量缓存的切片
        if not disable_tools:
            return self._tools_openai_cache
            
        # 只有在有过滤需求时才进行列表推导
        return [
            t for t in self._tools_openai_cache 
            if t["_original_name"] not in disable_tools
        ]

    async def get_tool_list(self):
        """获取简单的工具列表用于前端展示"""
        if not self._tools_cache:
            return []
        return [{"name": t.name, "description": t.description, "enabled": True} for t in self._tools_cache]

    async def call_tool(self, tool_name: str, tool_params: Dict[str, Any]) -> Any:
        # call_tool 必须加锁确保 session 存在
        async with self._lock:
            if not self._conn or not self._conn.session:
                return f"Error: MCP Server is not connected."
            try:
                return await self._conn.session.call_tool(tool_name, tool_params)
            except Exception as e:
                logging.error("Failed to call tool %s: %s", tool_name, e)
                return f"Failed to call tool {tool_name}: {e}"

# 保持 __main__ 用于测试
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    async def main():
        client = McpClient()
        await client.initialize(
            "test_server",
            {"command": "python", "args": ["-m", "mcp_server_time"]} # 示例配置
        )
        await asyncio.sleep(2) # 等待连接
        
        # 第一次获取（极快）
        print(await client.get_openai_functions())
        
        # 第二次获取（极快，无网络请求）
        print(await client.get_openai_functions())
        
        await client.close()
    
    # asyncio.run(main())