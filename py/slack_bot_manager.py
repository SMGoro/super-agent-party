import asyncio
import base64
import io
import logging
import re
import threading
import time
from typing import Dict, List, Optional, Any

import aiohttp
from slack_sdk.web.async_client import AsyncWebClient
from slack_sdk.socket_mode.aiohttp import SocketModeClient
from slack_sdk.socket_mode.request import SocketModeRequest
from slack_sdk.socket_mode.response import SocketModeResponse

from openai import AsyncOpenAI
from pydantic import BaseModel
from py.get_setting import get_port, load_settings

# ------------------ 配置模型 (严格对齐) ------------------
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
        
        self.bot_user_id: Optional[str] = None
        
        # --- 完全复刻 Discord 的状态存储 ---
        self.memory: Dict[str, List[dict]] = {}      # channel_id -> msgs
        self.async_tools: Dict[str, List[str]] = {}  # channel_id -> ids
        self.file_links: Dict[str, List[str]] = {}   # channel_id -> links

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
            raise RuntimeError("Slack 机器人启动超时")

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
        
        # 获取机器人 ID 用于防递归
        auth = await web_client.auth_test()
        self.bot_user_id = auth["user_id"]

        self.socket_client = SocketModeClient(app_token=config.app_token, web_client=web_client)

        async def process_listener(client, req: SocketModeRequest):
            if req.type == "events_api":
                await client.send_socket_mode_response(SocketModeResponse(envelope_id=req.envelope_id))
                event = req.payload.get("event", {})
                # 过滤逻辑
                if event.get("user") == self.bot_user_id or event.get("bot_id") or "subtype" in event:
                    return
                if event.get("type") in ["message", "app_mention"]:
                    asyncio.ensure_future(self._handle_message(event, web_client))

        self.socket_client.socket_mode_request_listeners.append(process_listener)
        await self.socket_client.connect()
        self.is_running = True
        self._ready_complete.set()
        while self.is_running: await asyncio.sleep(1)

    def stop_bot(self):
        self.is_running = False
        if self.socket_client:
            asyncio.run_coroutine_threadsafe(self.socket_client.close(), self.loop)
        if self.loop:
            self.loop.call_soon_threadsafe(self.loop.stop)
        self.is_running = False

    def get_status(self):
        return {"is_running": self.is_running}

    # ---------- 核心处理：1:1 复刻 Discord 逻辑 ----------
    async def _handle_message(self, event: dict, web_client: AsyncWebClient):
        cid = event["channel"]
        text = event.get("text", "").strip()

        if cid not in self.memory:
            self.memory[cid], self.async_tools[cid], self.file_links[cid] = [], [], []

        if self.config.wakeWord and self.config.wakeWord not in text: return

        if self.config.quick_restart and text in ["/重启", "/restart"]:
            self.memory[cid].clear()
            await web_client.chat_postMessage(channel=cid, text="对话记录已重置。")
            return

        self.memory[cid].append({"role": "user", "content": text})

        # --- 完全复刻 Discord 的 state 状态机 ---
        state = {
            "text_buffer": "", 
            "image_buffer": "", 
            "image_cache": [],
        }

        # 发送占位消息
        initial_resp = await web_client.chat_postMessage(channel=cid, text="...")
        reply_ts = initial_resp["ts"]

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

            full_response = []
            last_update_time = time.time()

            async for chunk in stream:
                if not chunk.choices: continue
                delta_raw = chunk.choices[0].delta

                # 1. 捕获工具和链接 (同 Discord)
                tool_link = getattr(delta_raw, "tool_link", None)
                if tool_link and settings.get("tools", {}).get("toolMemorandum", {}).get("enabled"):
                    if tool_link not in self.file_links[cid]: self.file_links[cid].append(tool_link)

                async_tool_id = getattr(delta_raw, "async_tool_id", None)
                if async_tool_id:
                    if async_tool_id not in self.async_tools[cid]: self.async_tools[cid].append(async_tool_id)
                    else: self.async_tools[cid].remove(async_tool_id)

                # 2. 文本解析 (同 Discord)
                content = delta_raw.content or ""
                reasoning = getattr(delta_raw, "reasoning_content", None) or ""
                if reasoning and self.config.reasoning_visible:
                    content = reasoning

                full_response.append(content)
                state["text_buffer"] += content
                state["image_buffer"] += content

                # 3. Slack 流式更新节流
                now = time.time()
                if (now - last_update_time > 1.2) or any(sep in content for sep in self.config.separators):
                    seg = self._clean_text(state["text_buffer"])
                    if seg and seg.strip():
                        await web_client.chat_update(channel=cid, ts=reply_ts, text=seg + " ▌")
                        last_update_time = now

            # 4. 最终文本更新
            full_content = "".join(full_response)
            final_text = self._clean_text(full_content)
            await web_client.chat_update(channel=cid, ts=reply_ts, text=final_text or "回复完成。")

            # 5. 图片提取与发送 (完全复刻 Discord 逻辑)
            self._extract_images(state)
            for img_url in state["image_cache"]:
                await self._send_image(cid, img_url, web_client)

            # 6. TTS 处理 (Payload 补全)
            if self.config.enable_tts:
                await self._send_voice(cid, full_content, web_client)

            # 7. 记忆管理 (同 Discord)
            self.memory[cid].append({"role": "assistant", "content": full_content})
            if self.config.memory_limit > 0:
                while len(self.memory[cid]) > self.config.memory_limit * 2:
                    self.memory[cid].pop(0)

        except Exception as e:
            logging.error(f"Slack Bot Error: {e}")
            await web_client.chat_update(channel=cid, ts=reply_ts, text=f"❌ 处理消息失败: {e}")

    # ---------- 工具函数 (1:1 复刻 Discord) ----------
    def _extract_images(self, state: Dict[str, Any]):
        pattern = r'!\[.*?\]\((https?://[^\s)]+)'
        for m in re.finditer(pattern, state["image_buffer"]):
            state["image_cache"].append(m.group(1))

    def _clean_text(self, text: str) -> str:
        return re.sub(r"!\[.*?\]\(.*?\)", "", text).strip()

    async def _send_image(self, cid: str, url: str, web_client: AsyncWebClient):
        try:
            async with aiohttp.ClientSession() as s:
                async with s.get(url) as r:
                    if r.status == 200:
                        data = await r.read()
                        await web_client.files_upload_v2(channel=cid, file=data, filename="image.png")
        except Exception as e:
            logging.error(f"发送图片失败: {e}")

    async def _send_voice(self, cid: str, text: str, web_client: AsyncWebClient):
        try:
            import aiohttp
            settings = await load_settings()
            tts_settings = settings.get("ttsSettings", {})
            
            clean_text = re.sub(r'[*_~`#]|!\[.*?\]\(.*?\)', '', text)
            if not clean_text.strip(): return

            # --- 优化点：针对 Slack 调整 Payload ---
            payload = {
                "text": clean_text[:300],
                "voice": "default",
                "ttsSettings": tts_settings,
                "index": 0,
                # Slack 建议关闭 mobile_optimized 以获取标准 mp3
                "mobile_optimized": False, 
                "format": "mp3" # 👈 改为 mp3，Slack 兼容性更高
            }

            async with aiohttp.ClientSession() as s:
                async with s.post(f"http://127.0.0.1:{get_port()}/tts", json=payload) as r:
                    if r.status == 200:
                        audio = await r.read()
                        
                        # 使用 v2 接口上传
                        await web_client.files_upload_v2(
                            channel=cid, 
                            file=audio, 
                            filename="voice.mp3", # 👈 扩展名改为 mp3
                            title="语音回复",       # 增加标题
                            initial_comment="🔊 语音合成已完成，点击上方文件名可试听。" # 引导用户
                        )
                    else:
                        logging.error(f"TTS 接口返回错误: {r.status}")
        except Exception as e:
            logging.error(f"Slack TTS 发送失败: {e}")