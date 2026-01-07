import json
import time
import requests
import asyncio
import websockets
from py.get_setting import load_settings

# 全局变量，用于保持当前上下文
CURRENT_PAGE_INDEX = 0

async def get_cdp_port():
    settings = await load_settings()
    # 默认回退到 9222，或者你的配置值
    return settings.get('chromeMCPSettings', {}).get('CDPport', 3456) # 假设你主进程默认端口是3456

async def get_targets():
    """获取所有 CDP 目标"""
    port = await get_cdp_port()
    try:
        resp = requests.get(f'http://127.0.0.1:{port}/json/list')
        return resp.json()
    except Exception as e:
        print(f"CDP Connection Error: {e}")
        return []

async def get_main_window_ws():
    """获取主窗口（Vue 控制器）的 WebSocket URL"""
    targets = await get_targets()
    
    # 调试：打印所有目标，方便你看清楚当前有哪些窗口
    # print("Current CDP Targets:", json.dumps(targets, indent=2))
    
    for t in targets:
        url = t.get('url', '')
        title = t.get('title', '')
        target_type = t.get('type')

        # 1. 必须是 page 类型 (排除 webview 标签页, service_worker 等)
        if target_type != 'page':
            continue

        # 2. ★ 关键：排除 VRM 窗口
        # VRM 窗口的 URL 通常包含 'vrm.html'
        if 'vrm.html' in url:
            continue
            
        # 3. ★ 关键：排除开发者工具窗口 (如果打开了 DevTools)
        if 'devtools://' in url:
            continue
            
        # 4. (可选) 排除扩展程序窗口
        if 'ext' in url:
            continue

        # 5. 找到主窗口
        # 主窗口的特征通常是：
        # - URL 包含 'skeleton.html' (骨架屏阶段)
        # - 或者 URL 是 'http://127.0.0.1:端口/' (加载完成阶段)
        # - 只要不是上面排除的特定窗口，剩下的 page 通常就是主窗口
        return t.get('webSocketDebuggerUrl')
        
    print("Error: Could not find Main Window in CDP targets.")
    return None

async def get_webview_ws(index=None):
    """获取具体网页的 WebSocket URL"""
    targets = await get_targets()
    # 过滤出所有 webview (实际的网页标签)
    webviews = [t for t in targets if t['type'] == 'webview']
    
    target_idx = index if index is not None else CURRENT_PAGE_INDEX
    
    if 0 <= target_idx < len(webviews):
        return webviews[target_idx].get('webSocketDebuggerUrl')
    return None

async def cdp_command(ws_url, method, params=None):
    """发送 CDP 命令的通用函数"""
    if not ws_url:
        return {"error": "Target not found"}
    
    async with websockets.connect(ws_url) as ws:
        cmd_id = 1
        message = {
            "id": cmd_id,
            "method": method,
            "params": params or {}
        }
        await ws.send(json.dumps(message))
        
        while True:
            response = await ws.recv()
            data = json.loads(response)
            if data.get('id') == cmd_id:
                return data.get('result', {})

# ==========================================
# Navigation Automation (控制 Vue)
# ==========================================

async def list_pages():
    """List currently open pages."""
    targets = await get_targets()
    webviews = [t for t in targets if t['type'] == 'webview']
    
    pages_info = []
    for idx, t in enumerate(webviews):
        pages_info.append({
            "index": idx,
            "id": t['id'],
            "title": t['title'],
            "url": t['url']
        })
    return json.dumps(pages_info, ensure_ascii=False)

async def new_page(url, timeout=0):
    """Create a new page via Vue."""
    ws_url = await get_main_window_ws()
    # 调用 Vue 的 openUrlInNewTab 方法
    # 注意：window.aiBrowser 是我们在 mounted 中挂载的
    expression = f"window.aiBrowser.openUrlInNewTab('{url}')"
    await cdp_command(ws_url, "Runtime.evaluate", {"expression": expression})
    
    # 简单等待加载，实际生产可能需要轮询 list_pages 确认
    if timeout > 0:
        await asyncio.sleep(timeout / 1000)
    return "Page creating initiated."

async def close_page(pageIdx):
    """Close a page by index."""
    ws_url = await get_main_window_ws()
    # 1. 获取 Tab ID
    # 我们需要在 JS 端执行逻辑，或者先 python 获取 ID 再传进去
    # 这里直接在 JS 端完成更方便
    js_script = f"""
    (function() {{
        const tabId = window.aiBrowser.getTabIdByIndex({pageIdx});
        if(tabId) {{
            window.aiBrowser.closeTab(tabId);
            return "Closed";
        }}
        return "Tab not found";
    }})()
    """
    res = await cdp_command(ws_url, "Runtime.evaluate", {"expression": js_script, "returnByValue": True})
    return res.get('value', 'Error executing close')

async def select_page(pageIdx, bringToFront=True):
    """Select a page context."""
    global CURRENT_PAGE_INDEX
    targets = await get_targets()
    webviews = [t for t in targets if t['type'] == 'webview']
    
    if 0 <= pageIdx < len(webviews):
        CURRENT_PAGE_INDEX = pageIdx
        
        if bringToFront:
            ws_url = await get_main_window_ws()
            # 调用 Vue 切换 Tab
            js_script = f"""
            (function() {{
                const tabId = window.aiBrowser.getTabIdByIndex({pageIdx});
                if(tabId) window.aiBrowser.switchTab(tabId);
            }})()
            """
            await cdp_command(ws_url, "Runtime.evaluate", {"expression": js_script})
        
        return f"Selected page index {pageIdx}"
    return "Page index out of range"

async def navigate_page(url, type="url", ignoreCache=False, timeout=0):
    """Navigate current page."""
    ws_url = await get_webview_ws() # 连接当前选中的 Webview
    
    if type == "url":
        await cdp_command(ws_url, "Page.navigate", {"url": url})
    elif type == "reload":
        await cdp_command(ws_url, "Page.reload", {"ignoreCache": ignoreCache})
    elif type == "back":
        # 获取 History
        hist = await cdp_command(ws_url, "Page.getNavigationHistory")
        idx = hist.get('currentIndex')
        if idx > 0:
            entry_id = hist['entries'][idx-1]['id']
            await cdp_command(ws_url, "Page.navigateToHistoryEntry", {"entryId": entry_id})
    elif type == "forward":
        hist = await cdp_command(ws_url, "Page.getNavigationHistory")
        idx = hist.get('currentIndex')
        entries = hist.get('entries', [])
        if idx < len(entries) - 1:
            entry_id = entries[idx+1]['id']
            await cdp_command(ws_url, "Page.navigateToHistoryEntry", {"entryId": entry_id})
            
    return f"Navigated {type}"

# ==========================================
# Input Automation (Via Vue Controller)
# ==========================================

async def call_vue_method(method_name, args_list=None):
    """
    通用函数：调用 window.aiBrowser 的方法
    """
    ws_url = await get_main_window_ws()
    if not ws_url:
        return {"error": "Main window not found"}

    # 构造参数字符串
    if args_list:
        json_args = [json.dumps(arg) for arg in args_list]
        args_str = ", ".join(json_args)
    else:
        args_str = ""

    # 调用 Vue 方法
    expression = f"window.aiBrowser.{method_name}({args_str})"
    
    res = await cdp_command(ws_url, "Runtime.evaluate", {
        "expression": expression,
        "returnByValue": True, 
        "awaitPromise": True   # 必须等待 Vue 的 async 方法完成
    })
    
    # --- 错误处理 ---
    if 'exceptionDetails' in res:
        # 提取详细错误信息
        exc = res['exceptionDetails']
        msg = exc.get('text', 'Unknown Error')
        if 'exception' in exc and 'description' in exc['exception']:
            msg = f"{msg}: {exc['exception']['description']}"
        return f"Error executing {method_name}: {msg}"
    
    # --- 关键修正：解析嵌套的 result 对象 ---
    # CDP 返回格式: { "result": { "type": "string", "value": "..." } }
    remote_object = res.get('result', {})
    
    if 'value' in remote_object:
        return remote_object['value']
    
    # 如果返回的是 undefined (例如函数没有 return)，value 字段不存在
    if remote_object.get('type') == 'undefined':
        return "Success (No content returned)"
        
    return f"Operation completed (Type: {remote_object.get('type')})"

# ------------------------------------------
# Interaction Tools (Complete List)
# ------------------------------------------

async def take_snapshot(filePath=None, verbose=False):
    """
    获取页面可交互元素的 DOM 树快照。
    """
    # 调用 Vue 方法生成快照字符串
    result = await call_vue_method('getWebviewSnapshot', [verbose])
    
    # 如果指定了 filePath，则保存到文件（模拟 Agent 行为）
    if filePath and result and isinstance(result, str):
        try:
            with open(filePath, 'w', encoding='utf-8') as f:
                f.write(result)
            return f"Snapshot saved to {filePath}"
        except Exception as e:
            return f"Error saving snapshot: {str(e)}"
            
    # 否则直接返回快照内容
    return result

async def click(uid, dblClick=False):
    """点击元素"""
    return await call_vue_method('webviewClick', [uid, dblClick])

async def fill(uid, value):
    """填写输入框"""
    return await call_vue_method('webviewFill', [uid, value])

async def fill_form(elements):
    """
    批量填写表单
    elements: [{'uid': '...', 'value': '...'}, ...]
    """
    return await call_vue_method('webviewFillForm', [elements])

async def drag(from_uid, to_uid):
    """拖拽元素"""
    return await call_vue_method('webviewDrag', [from_uid, to_uid])

async def handle_dialog(action, promptText=None):
    """处理弹窗 (alert/confirm/prompt)"""
    return await call_vue_method('webviewHandleDialog', [action, promptText])

async def hover(uid):
    """悬停"""
    return await call_vue_method('webviewHover', [uid])

async def press_key(key):
    """按键"""
    return await call_vue_method('webviewPressKey', [key])

async def upload_file(uid, filePath):
    """上传文件"""
    # 这是一个复杂操作，需要 Vue 端配合 debugger 或 remote
    return await call_vue_method('webviewUploadFile', [uid, filePath])

# ------------------------------------------
# Navigation Tools
# ------------------------------------------

async def list_pages():
    """列出所有标签页"""
    return await call_vue_method('getPagesInfo')

async def new_page(url, timeout=0):
    """新建标签页"""
    return await call_vue_method('openUrlInNewTab', [url])

async def close_page(pageIdx):
    """关闭标签页"""
    return await call_vue_method('closeTabByIndex', [pageIdx])

async def select_page(pageIdx, bringToFront=True):
    """选择/切换标签页"""
    return await call_vue_method('switchTabByIndex', [pageIdx])

async def navigate_page(type="url", url=None, ignoreCache=False, timeout=0):
    """页面导航"""
    return await call_vue_method('browserNavigate', [type, url, ignoreCache])

async def wait_for(text, timeout=5000):
    """等待文本出现"""
    return await call_vue_method('webviewWaitFor', [text, timeout])

# ------------------------------------------
# Debugging Tools
# ------------------------------------------

async def evaluate_script(function, args=None):
    """执行 JS"""
    return await call_vue_method('executeInActiveWebview', [function, args or []])

async def take_screenshot(filePath=None, format="png", fullPage=False, quality=None, uid=None):
    """
    截图。注意：Electron Webview 的 capturePage 主要是视口截图。
    """
    # Vue 端返回的是 Base64
    base64_data = await call_vue_method('captureWebviewScreenshot', [fullPage, uid])
    
    if base64_data.startswith("Error"):
        return base64_data

    # 如果需要保存到文件
    if filePath:
        try:
            import base64
            # 去掉 data:image/png;base64, 前缀
            if ',' in base64_data:
                base64_data = base64_data.split(',')[1]
            
            with open(filePath, "wb") as f:
                f.write(base64.b64decode(base64_data))
            return f"Screenshot saved to {filePath}"
        except Exception as e:
            return f"Error saving screenshot: {str(e)}"
            
    return "Screenshot captured (Base64 data hidden)"


# ==========================================
# Tool Definitions (JSON Schemas)
# ==========================================

all_cdp_tools = [
    # --- Navigation ---
    {
        "type": "function",
        "function": {
            "name": "list_pages",
            "description": "Get a list of pages open in the browser.",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "new_page",
            "description": "Creates a new page in the browser tab bar.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to load"},
                    "timeout": {"type": "integer"}
                },
                "required": ["url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "close_page",
            "description": "Closes the page by its index.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pageIdx": {"type": "integer", "description": "Index of the page to close"}
                },
                "required": ["pageIdx"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "select_page",
            "description": "Switch tab to the specified page index.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pageIdx": {"type": "integer"},
                    "bringToFront": {"type": "boolean"}
                },
                "required": ["pageIdx"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "navigate_page",
            "description": "Navigates the currently selected page.",
            "parameters": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": ["url", "back", "forward", "reload"]},
                    "url": {"type": "string"},
                    "ignoreCache": {"type": "boolean"}
                },
                "required": ["type"]
            }
        }
    },
    
    # --- Debugging & Input ---
    {
        "type": "function",
        "function": {
            "name": "take_snapshot",
            "description": "Get the accessibility tree of the current page to find UIDs for interaction.",
            "parameters": {
                "type": "object",
                "properties": {
                    "verbose": {"type": "boolean"}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "click",
            "description": "Clicks an element identified by UID from take_snapshot.",
            "parameters": {
                "type": "object",
                "properties": {
                    "uid": {"type": "string", "description": "The BackendNodeId from snapshot"},
                    "dblClick": {"type": "boolean"}
                },
                "required": ["uid"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "fill",
            "description": "Type text into an input element.",
            "parameters": {
                "type": "object",
                "properties": {
                    "uid": {"type": "string"},
                    "value": {"type": "string"}
                },
                "required": ["uid", "value"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "evaluate_script",
            "description": "Run JS in the current page.",
            "parameters": {
                "type": "object",
                "properties": {
                    "function": {"type": "string", "description": "JS function body"},
                    "args": {"type": "array"}
                },
                "required": ["function"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "hover",
            "description": "Hover over an element identified by UID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "uid": {"type": "string", "description": "The BackendNodeId from snapshot"}
                },
                "required": ["uid"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "press_key",
            "description": "Press a key or key combination (e.g. 'Enter', 'Control+a', 'ArrowDown').",
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {"type": "string", "description": "The key combination to press."}
                },
                "required": ["key"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "wait_for",
            "description": "Wait for specific text to appear on the page.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "The text to wait for."},
                    "timeout": {"type": "integer", "description": "Timeout in milliseconds (default 1000).","minimum": 0,"default": 1000,"maximum": 5000}
                },
                "required": ["text"]
            }
        }
    }
]