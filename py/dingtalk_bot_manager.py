import asyncio
import json
import threading
import os
import time
import logging
import aiohttp
import re
import base64
from typing import Optional, List, Dict, Any
from pydantic import BaseModel
from openai import AsyncOpenAI

# 钉钉官方 SDK
import dingtalk_stream
from dingtalk_stream import AckMessage, ChatbotMessage

# 假设这两个函数在你的 py.get_setting 中定义
from py.get_setting import get_port, load_settings

# 配置模型
class DingtalkBotConfig(BaseModel):
    DingtalkAgent: str
    memoryLimit: int
    appKey: str
    appSecret: str
    separators: List[str]
    reasoningVisible: bool
    quickRestart: bool
    enableTTS: bool 
    wakeWord: str

class DingtalkBotManager:
    def __init__(self):
        self.bot_thread: Optional[threading.Thread] = None
        self.is_running = False
        self.config = None
        self._startup_error = None
        self.client = None
        
    def start_bot(self, config: DingtalkBotConfig):
        if self.is_running:
            raise Exception("钉钉机器人已在运行")
        self.config = config
        self._startup_error = None
        self.bot_thread = threading.Thread(target=self._run_bot_thread, args=(config,), daemon=True)
        self.bot_thread.start()
        self.is_running = True

    def _run_bot_thread(self, config):
        try:
            self.bot_logic = DingtalkClientLogic(config)
            credential = dingtalk_stream.Credential(config.appKey, config.appSecret)
            self.client = dingtalk_stream.DingTalkStreamClient(credential)
            
            handler = DingtalkInternalHandler(self.bot_logic)
            self.client.register_callback_handler(ChatbotMessage.TOPIC, handler)
            
            logging.info("钉钉 AI 机器人已上线")
            self.client.start_forever()
        except Exception as e:
            self._startup_error = str(e)
            logging.error(f"钉钉机器人线程异常: {e}")
        finally:
            self.is_running = False

    def stop_bot(self):
        if self.client:
            try: self.client.stop()
            except: pass
        self.is_running = False

class DingtalkInternalHandler(dingtalk_stream.ChatbotHandler):
    def __init__(self, bot_logic):
        super(DingtalkInternalHandler, self).__init__()
        self.bot_logic = bot_logic

    async def process(self, callback: dingtalk_stream.CallbackMessage):
        try:
            # 解析原始消息
            incoming_message = ChatbotMessage.from_dict(callback.data)
            # 传入完整数据 callback.data 以便解析更多隐藏字段
            await self.bot_logic.on_message(callback.data, incoming_message, self)
        except Exception as e:
            logging.error(f"消息处理异常: {e}")
        return AckMessage.STATUS_OK, 'OK'

class DingtalkClientLogic:
    def __init__(self, config):
        self.config = config
        self.memoryList = {}
        self.port = get_port()
        self.separators = config.separators if config.separators else ['。', '\n', '？', '！']

    async def _get_image_base64(self, url: str) -> Optional[str]:
        """下载钉钉图片并转换为 Base64，解决 AI 访问 403 问题"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=10) as response:
                    if response.status == 200:
                        data = await response.read()
                        return base64.b64encode(data).decode('utf-8')
                    else:
                        logging.error(f"图片下载失败, HTTP状态码: {response.status}")
        except Exception as e:
            logging.error(f"图片处理异常: {e}")
        return None

    async def on_message(self, raw_data: dict, incoming_message: ChatbotMessage, handler: DingtalkInternalHandler):
        cid = incoming_message.conversation_id
        msg_type = incoming_message.message_type
        
        user_text = ""
        user_content_items = []
        has_image = False

        # --- A. 增强型消息解析 ---
        
        # 提取文字 (无论什么类型，先看 text 字段)
        if hasattr(incoming_message, 'text') and incoming_message.text:
            user_text = incoming_message.text.content.strip()

        # 处理图片
        if msg_type == "picture":
            download_code = incoming_message.image_content.download_code
            if download_code:
                # 获取钉钉内部临时下载地址
                img_url = handler.get_image_download_url(download_code)
                if img_url:
                    base64_str = await self._get_image_base64(img_url)
                    if base64_str:
                        has_image = True
                        user_content_items.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_str}"}
                        })
            
            # 钉钉在 picture 类型消息里，文字可能藏在 raw_data 中
            if not user_text:
                user_text = raw_data.get("content", {}).get("text", "").strip()

        # 处理富文本
        elif msg_type == "richText":
            if hasattr(incoming_message, 'rich_text') and incoming_message.rich_text:
                user_text = incoming_message.rich_text.text.strip()

        # --- B. 指令与过滤 ---
        if not user_text and not has_image:
            return

        # 快速重启指令
        if self.config.quickRestart and user_text and ("/重启" in user_text or "/restart" in user_text):
            self.memoryList[cid] = []
            handler.reply_text("对话记录已重置。", incoming_message)
            return
        
        # 唤醒词检查 (如果是图片，通常默认允许处理，或者你也可以加上唤醒词限制)
        if self.config.wakeWord and self.config.wakeWord not in user_text and not has_image:
            return

        # --- C. 构造 OpenAI 消息格式 ---
        if cid not in self.memoryList: 
            self.memoryList[cid] = []
        
        # 构造当前轮次的内容
        current_content = []
        if user_text:
            current_content.append({"type": "text", "text": user_text})
        if has_image:
            current_content.extend(user_content_items)
            # 如果有图无字，补一个引导语
            if not user_text:
                current_content.insert(0, {"type": "text", "text": "请分析这张图片"})

        self.memoryList[cid].append({"role": "user", "content": current_content})

        # --- D. AI 调用与流式输出 ---
        ai_client = AsyncOpenAI(api_key="none", base_url=f"http://127.0.0.1:{self.port}/v1")
        state = {"text_buffer": "", "full_response": ""}
        
        try:
            stream = await ai_client.chat.completions.create(
                model=self.config.DingtalkAgent,
                messages=self.memoryList[cid],
                stream=True
            )

            async for chunk in stream:
                if not chunk.choices: continue
                delta = chunk.choices[0].delta
                
                # 处理推理内容 (如 DeepSeek R1)
                reasoning = ""
                if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                    if self.config.reasoningVisible:
                        reasoning = delta.reasoning_content
                
                content = delta.content or ""
                combined_chunk = reasoning + content
                
                if not combined_chunk:
                    continue

                state["text_buffer"] += combined_chunk
                state["full_response"] += content

                # 检查分段符，流式回复钉钉
                if any(sep in state["text_buffer"] for sep in self.separators):
                    if state["text_buffer"].strip():
                        handler.reply_markdown("AI 助手", state["text_buffer"], incoming_message)
                    state["text_buffer"] = ""

            # 扫尾
            if state["text_buffer"].strip():
                handler.reply_markdown("AI 助手", state["text_buffer"], incoming_message)

            # --- E. 记忆持久化与裁剪 ---
            self.memoryList[cid].append({"role": "assistant", "content": state["full_response"]})
            if self.config.memoryLimit > 0:
                # 保持 memoryLimit 组对话 (1 user + 1 assistant = 2条)
                while len(self.memoryList[cid]) > self.config.memoryLimit * 2:
                    self.memoryList[cid].pop(0)

        except Exception as e:
            logging.error(f"钉钉 AI 生成异常: {e}")
            handler.reply_text(f"抱歉，处理消息时出错: {str(e)}", incoming_message)