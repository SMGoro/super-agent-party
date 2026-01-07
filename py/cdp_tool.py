from py.get_setting import load_settings
import requests

async def list_page():
    """List all pages"""
    settings = await load_settings()
    chromeMCPSettings = settings.get('chromeMCPSettings')
    if chromeMCPSettings:
        CDPport = chromeMCPSettings.get('CDPport',9222)
        url = f'http://127.0.0.1:{CDPport}/json'
        response = requests.get(url)
        if response.status_code == 200:
            # 解析返回的json数据
            data = response.json()
            # 只保留每个"type": "webview"的页面的url和id字段
            data = [item for item in data if item.get("type") == "webview"]
            data = [(item.get("url"), item.get("id")) for item in data]
        if data:
            return f"当前所有页面：{data}"
        else:
            return "当前浏览器内没有打开的页面"
    else:
        return "未找到chromeMCPSettings设置"


list_page_tool = {
    "type": "function",
    "function": {
        "name": "list_page",
        "description": f"查看当前所有页面url和id信息",
        "parameters": {
            "type": "object",
            "properties": {
            },
            "required": [],
        },
    },
}

all_cdp_tools = [
    list_page_tool
]