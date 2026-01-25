import asyncio
import base64
import io
import logging
import re
import threading
import time
from typing import Dict, List, Optional, Any

# 正确的导入路径和类名
from slack_sdk.web.async_client import AsyncWebClient
from slack_sdk.socket_mode.aiohttp import SocketModeClient
from slack_sdk.socket_mode.request import SocketModeRequest
from slack_sdk.socket_mode.response import SocketModeResponse

from openai import AsyncOpenAI
from pydantic import BaseModel
from py.get_setting import get_port, load_settings

# ------------------ 配置模型 (严格对齐前端) ------------------
class SlackBotConfig(BaseModel):
    bot_token: str
    app_token: str
    llm_model: str = "super-model"
    memory_limit: int = 30
    separators: List[str] = ["。", "\n", "？", "！"]
    reasoning_visible: bool = True
    quick_restart: bool = True
    enable_tts: bool = False
    wakeWord: str = ""

# ------------------ Slack 机器人管理器 ------------------
class SlackBotManager:
    def __init__(self):
        self.bot_thread: Optional[threading.Thread] = None
        self.socket_client: Optional[SocketModeClient] = None
        self.is_running = False
        self.config: Optional[SlackBotConfig] = None
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self._ready_complete = threading.Event()
        
        # 机器人身份标识，用于防递归 loop
        self.bot_user_id: Optional[str] = None
        
        # --- 完全复刻 Discord 的状态存储 ---
        self.memory: Dict[str, List[dict]] = {}      # channel_id -> msgs
        self.async_tools: Dict[str, List[str]] = {}  # channel_id -> async_tool_ids
        self.file_links: Dict[str, List[str]] = {}   # channel_id -> file_links

    def start_bot(self, config: SlackBotConfig):
        if self.is_running:
            raise RuntimeError("Slack 机器人已在运行")
        self.config = config
        self._ready_complete.clear()

        self.bot_thread = threading.Thread(
            target=self._run_bot_thread, args=(config,), daemon=True, name="SlackBotThread"
        )
        self.bot_thread.start()

        if not self._ready_complete.wait(timeout=30):
            self.stop_bot()
            raise RuntimeError("Slack 机器人启动超时，请检查网络和 Token")

    def _run_bot_thread(self, config: SlackBotConfig):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        try:
            self.loop.run_until_complete(self._async_start(config))
        except Exception as e:
            logging.exception(f"Slack 运行异常: {e}")
        finally:
            self.is_running = False

    async def _async_start(self, config: SlackBotConfig):
        web_client = AsyncWebClient(token=config.bot_token)
        
        # --- 获取机器人自己的 User ID (防止自己回复自己) ---
        try:
            auth_info = await web_client.auth_test()
            self.bot_user_id = auth_info["user_id"]
            logging.info(f"✅ Slack 身份验证成功: {auth_info['user']} (ID: {self.bot_user_id})")
        except Exception as e:
            logging.error(f"❌ Slack 身份验证失败: {e}")
            raise e

        # 初始化 SocketModeClient
        self.socket_client = SocketModeClient(
            app_token=config.app_token, 
            web_client=web_client
        )

        async def process_listener(client: SocketModeClient, req: SocketModeRequest):
            if req.type == "events_api":
                # 使用异步正确的方法发送响应
                await client.send_socket_mode_response(SocketModeResponse(envelope_id=req.envelope_id))
                
                payload = req.payload
                event = payload.get("event", {})
                event_type = event.get("type")
                user_id = event.get("user")

                # --- 严格过滤：防止递归和处理无效事件 ---
                if user_id == self.bot_user_id or event.get("bot_id") or "subtype" in event:
                    return

                # 判定消息来源
                is_mention = (event_type == "app_mention")
                is_dm = (event_type == "message" and event.get("channel_type") == "im")
                is_channel_msg = (event_type == "message" and "channel_type" in event)

                if is_mention or is_dm or is_channel_msg:
                    # 异步启动 LLM 处理
                    asyncio.ensure_future(self._handle_message(event, web_client))

        self.socket_client.socket_mode_request_listeners.append(process_listener)
        
        await self.socket_client.connect()
        self.is_running = True
        self._ready_complete.set()
        
        while self.is_running:
            await asyncio.sleep(1)

    def stop_bot(self):
        self.is_running = False
        if self.socket_client:
            asyncio.run_coroutine_threadsafe(self.socket_client.close(), self.loop)
        if self.loop:
            self.loop.call_soon_threadsafe(self.loop.stop)
        if self.bot_thread:
            self.bot_thread.join(timeout=5)

    def get_status(self):
        return {"is_running": self.is_running, "config": self.config.model_dump() if self.config else None}

    # ---------- 核心处理逻辑 (流式主界面版) ----------
    async def _handle_message(self, event: dict, web_client: AsyncWebClient):
        cid = event["channel"]
        text = event.get("text", "").strip()
        ts = event["ts"]

        # 1. 状态初始化 (同 Discord)
        if cid not in self.memory:
            self.memory[cid] = []
            self.async_tools[cid] = []
            self.file_links[cid] = []

        # 2. 唤醒词逻辑
        if self.config.wakeWord and self.config.wakeWord not in text:
            return

        # 3. 快速重启逻辑
        if self.config.quick_restart and text in ["/重启", "/restart"]:
            self.memory[cid].clear()
            self.async_tools[cid].clear()
            self.file_links[cid].clear()
            # 这里直接发在主频道
            await web_client.chat_postMessage(channel=cid, text="对话记录已重置。")
            return

        # 4. 组装内容
        self.memory[cid].append({"role": "user", "content": text})

        # 5. 发送思考中占位符 (移除了 thread_ts，确保在主界面流式显示)
        try:
            initial_resp = await web_client.chat_postMessage(channel=cid, text="...")
            reply_ts = initial_resp["ts"]
        except Exception as e:
            logging.error(f"发送初始消息失败: {e}")
            return

        # 6. 请求 LLM (参数与 Discord 一模一样)
        settings = await load_settings()
        client_ai = AsyncOpenAI(api_key="super-secret-key", base_url=f"http://127.0.0.1:{get_port()}/v1")

        try:
            stream = await client_ai.chat.completions.create(
                model=self.config.llm_model,
                messages=self.memory[cid],
                stream=True,
                extra_body={
                    "asyncToolsID": self.async_tools[cid],
                    "fileLinks": self.file_links[cid],
                    "is_app_bot": True,
                },
            )

            full_response_list = []
            text_buffer = ""
            last_update_time = time.time()

            async for chunk in stream:
                if not chunk.choices: continue
                delta_raw = chunk.choices[0].delta

                # --- 工具、链接捕获 (1:1 复刻 Discord) ---
                async_tool_id = getattr(delta_raw, "async_tool_id", None)
                tool_link = getattr(delta_raw, "tool_link", None)

                if tool_link and settings.get("tools", {}).get("toolMemorandum", {}).get("enabled"):
                    if tool_link not in self.file_links[cid]:
                        self.file_links[cid].append(tool_link)

                if async_tool_id:
                    if async_tool_id not in self.async_tools[cid]:
                        self.async_tools[cid].append(async_tool_id)
                    else:
                        self.async_tools[cid].remove(async_tool_id)

                # --- 文本流解析 ---
                reasoning = getattr(delta_raw, "reasoning_content", None) or ""
                content = delta_raw.content or ""
                
                if reasoning and self.config.reasoning_visible:
                    content = reasoning
                
                full_response_list.append(content)
                text_buffer += content

                # --- Slack 专属流式节流 (1.2秒/次，避免 429) ---
                now = time.time()
                if (now - last_update_time > 1.2) or any(sep in content for sep in self.config.separators):
                    clean_display = self._clean_text(text_buffer)
                    # 只有当清理后的文本不为空时，才执行 update，防止 no_text 错误
                    if clean_display and clean_display.strip():
                        await web_client.chat_update(channel=cid, ts=reply_ts, text=clean_display + " ▌")
                        last_update_time = now

            # 7. 最终更新 (主界面更新)
            full_content = "".join(full_response_list)
            final_display = self._clean_text(full_content)
            if final_display and final_display.strip():
                await web_client.chat_update(channel=cid, ts=reply_ts, text=final_display)
            else:
                await web_client.chat_update(channel=cid, ts=reply_ts, text="回复完成。")

            # 8. 图片处理 (不使用 thread_ts，直接发在主频道)
            img_urls = re.findall(r'!\[.*?\]\((https?://[^\s)]+)\)', full_content)
            for url in img_urls:
                await self._send_image(cid, url, web_client)

            # 9. TTS 处理 (不使用 thread_ts)
            if self.config.enable_tts:
                await self._send_voice(cid, full_content, web_client)

            # 10. 记忆更新与限制 (1:1 复刻 Discord)
            self.memory[cid].append({"role": "assistant", "content": full_content})
            if self.config.memory_limit > 0:
                while len(self.memory[cid]) > self.config.memory_limit * 2:
                    self.memory[cid].pop(0)

        except Exception as e:
            logging.error(f"Slack Bot LLM Error: {e}")
            # 如果报错，在频道里反馈，同样不带 thread_ts
            await web_client.chat_update(channel=cid, ts=reply_ts, text=f"❌ 处理消息失败：{str(e)}")

    def _clean_text(self, text: str) -> str:
        # 复刻 Discord 逻辑：移除 Markdown 图片链接，防止重复显示
        return re.sub(r"!\[.*?\]\(.*?\)", "", text).strip()

    async def _send_image(self, cid: str, url: str, web_client: AsyncWebClient):
        try:
            import aiohttp
            async with aiohttp.ClientSession() as s:
                async with s.get(url) as r:
                    if r.status == 200:
                        data = await r.read()
                        # files_upload_v2 不带 thread_ts，就会直接发在频道里
                        await web_client.files_upload_v2(channel=cid, file=data, filename="image.png")
        except Exception: pass

    async def _send_voice(self, cid: str, text: str, web_client: AsyncWebClient):
        try:
            import aiohttp
            # 清理 Markdown 后生成 TTS
            clean_text = re.sub(r'[*_~`#]|!\[.*?\]\(.*?\)', '', text)
            async with aiohttp.ClientSession() as s:
                async with s.post(f"http://127.0.0.1:{get_port()}/tts", json={"text": clean_text[:300], "format": "opus"}) as r:
                    if r.status == 200:
                        audio = await r.read()
                        # files_upload_v2 不带 thread_ts
                        await web_client.files_upload_v2(channel=cid, file=audio, filename="voice.opus")
        except Exception: pass