import json
import os
import time
import uuid
import requests
import asyncio
import websockets
from py.get_setting import UPLOAD_FILES_DIR, get_port, load_settings

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
    
    # 修改这里：增加 max_size 参数
    # 设置为 None 表示不限制大小，或者设置为 10 * 1024 * 1024 (10MB)
    async with websockets.connect(ws_url, max_size=10 * 1024 * 1024) as ws:
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

async def press_key(key,uid):
    """按键"""
    return await call_vue_method('webviewPressKey', [key, uid])

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

async def wait_for(text, timeout=1000):
    """等待文本出现"""
    return await call_vue_method('webviewWaitFor', [text, timeout])

# ------------------------------------------
# Debugging Tools
# ------------------------------------------

async def evaluate_script(function, args=None):
    """执行 JS"""
    return await call_vue_method('executeInActiveWebview', [function, args or []])

async def take_screenshot(fullPage=False, uid=None):
    """
    截图
    Vue 端已将图片保存到 uploaded_files 目录，并返回了 URL。
    """
    # 直接调用，返回值就是 URL (例如: http://127.0.0.1:3456/uploaded_files/xxx.jpg)
    result = await call_vue_method('captureWebviewScreenshot', [fullPage, uid])
    
    # 简单的错误检查
    if not result or result.startswith("Error") or result.startswith("Screenshot Error"):
        return f"Failed to capture screenshot: {result}"
        
    return f"Screenshot saved to {result}"

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
                    "args": {"type": "array","items": {"type": "string"}}
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
                    "key": {"type": "string", "description": "The key combination to press."},
                    "uid": {"type": "string", "description": "The BackendNodeId from snapshot"}
                },
                "required": ["key","uid"]
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
                    "timeout": {"type": "integer", "description": "Timeout in milliseconds (default 1000).","minimum": 100,"default": 1000,"maximum": 10000}
                },
                "required": ["text"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "take_screenshot",
            "description": "Take a screenshot of the current page.",
            "parameters": {
                "type": "object",
                "properties": {
                    "fullPage": {"type": "boolean", "description": "If set to true takes a screenshot of the full page instead of the currently visible viewport. Incompatible with uid."},
                    "uid": {"type": "string", "description": "The uid of an element on the page from the page content snapshot. If omitted takes a pages screenshot."}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "fill_form",
            "description": "Fill multiple form fields at once efficiently.",
            "parameters": {
                "type": "object",
                "properties": {
                    "elements": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "uid": {"type": "string", "description": "The UID of the input element."},
                                "value": {"type": "string", "description": "The value to fill."}
                            },
                            "required": ["uid", "value"]
                        },
                        "description": "List of elements to fill, e.g., [{'uid': 'ai-1', 'value': 'john'}, ...]"
                    }
                },
                "required": ["elements"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "drag",
            "description": "Drag an element from one position to another using UIDs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "from_uid": {"type": "string", "description": "The UID of the element to drag."},
                    "to_uid": {"type": "string", "description": "The UID of the target element to drop onto."}
                },
                "required": ["from_uid", "to_uid"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "handle_dialog",
            "description": "Handle JavaScript dialogs (alert, confirm, prompt).",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["accept", "dismiss"], "description": "Whether to accept or dismiss the dialog."},
                    "promptText": {"type": "string", "description": "Text to enter if the dialog is a prompt."}
                },
                "required": ["action"]
            }
        }
    }
]