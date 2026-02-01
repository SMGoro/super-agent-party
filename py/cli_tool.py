#!/usr/bin/env python3
import asyncio
import os
import re
import shutil
import subprocess
import json
import platform
import uuid
import tempfile
from pathlib import Path
from typing import AsyncIterator
from datetime import datetime
import aiofiles
import aiofiles.os
import glob as std_glob
import fnmatch

def get_shell_environment():
    """通过子进程获取完整的 shell 环境"""
    shell = os.environ.get('SHELL', '/bin/zsh')
    home = Path.home()
    
    config_commands = [
        f'source {home}/.zshrc && env',
        f'source {home}/.bash_profile && env', 
        f'source {home}/.bashrc && env',
        'env'
    ]
    
    for cmd in config_commands:
        try:
            result = subprocess.run(
                [shell, '-i', '-c', cmd],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    if '=' in line:
                        var_name, var_value = line.split('=', 1)
                        os.environ[var_name] = var_value
                print("Successfully loaded environment from shell")
                return
        except Exception as e:
            print(f"Failed to load environment with command '{cmd}': {e}")
            continue
    
    print("Warning: Could not load shell environment, using current environment")

get_shell_environment()

import anyio
from claude_agent_sdk import query, ClaudeAgentOptions, AssistantMessage, TextBlock
from py.get_setting import load_settings

# ==================== 公共工具函数 ====================

async def read_stream(stream, *, is_error: bool = False):
    """读取流并添加错误前缀"""
    if stream is None:
        return
    async for line in stream:
        prefix = "[ERROR] " if is_error else ""
        yield f"{prefix}{line.decode('utf-8', errors='replace').rstrip()}"

async def _merge_streams(*streams):
    """合并多个异步流"""
    streams = [s.__aiter__() for s in streams]
    while streams:
        for stream in list(streams):
            try:
                item = await stream.__anext__()
                yield item
            except StopAsyncIteration:
                streams.remove(stream)

# ==================== Docker Sandbox 基础设施 ====================

import hashlib

def get_safe_container_name(cwd: str) -> str:
    """
    根据路径生成合法容器名
    规则：sandbox- + 路径MD5前12位（确保唯一且合法）
    """
    abs_path = str(Path(cwd).resolve())
    path_hash = hashlib.md5(abs_path.encode()).hexdigest()[:12]
    return f"sandbox-{path_hash}"

async def get_or_create_docker_sandbox(cwd: str, image_name: str = "docker/sandbox-templates:latest") -> str:
    """
    获取或创建基于路径的持久化沙盒
    返回: 容器名（同时也是沙盒ID）
    """
    container_name = get_safe_container_name(cwd)
    
    check_proc = await asyncio.create_subprocess_exec(
        "docker", "ps", "-a", "--filter", f"name=^/{container_name}$", "--format", "{{.Names}}|{{.Status}}",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    stdout, _ = await check_proc.communicate()
    output = stdout.decode().strip()
    
    if container_name in output:
        status = output.split("|")[-1] if "|" in output else ""
        
        if "Up" in status:
            print(f"[INFO] 使用已运行的沙盒: {container_name}")
            return container_name
        else:
            print(f"[INFO] 启动已存在的沙盒: {container_name}")
            start_proc = await asyncio.create_subprocess_exec(
                "docker", "start", container_name,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            _, stderr = await start_proc.communicate()
            if start_proc.returncode != 0:
                raise Exception(f"启动沙盒失败: {stderr.decode()}")
            return container_name
    
    print(f"[INFO] 创建新沙盒: {container_name} (路径: {cwd})")
    
    create_cmd = [
        "docker", "run", "-d",
        "--name", container_name,
        "-v", f"{cwd}:/workspace",
        "-w", "/workspace",
        "--restart", "unless-stopped",
        "--label", f"sandbox.path={cwd}",
        "--label", "sandbox.type=persistent",
        image_name,
        "tail", "-f", "/dev/null"
    ]
    
    proc = await asyncio.create_subprocess_exec(
        *create_cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await proc.communicate()
    
    if proc.returncode == 0:
        container_id = stdout.decode().strip()[:12]
        print(f"[INFO] 沙盒创建成功: {container_id}")
        return container_name
    else:
        error_msg = stderr.decode()
        if "is already in use by container" in error_msg:
            await asyncio.sleep(0.5)
            return await get_or_create_docker_sandbox(cwd, image_name)
        raise Exception(f"创建沙盒失败: {error_msg}")

async def _exec_docker_cmd_simple(cwd: str, cmd_list: list) -> str:
    """
    内部辅助函数：在容器内执行简单命令并获取一次性输出
    """
    container_name = await get_or_create_docker_sandbox(cwd)
    
    full_cmd = ["docker", "exec", "-w", "/workspace", container_name] + cmd_list
    
    proc = await asyncio.create_subprocess_exec(
        *full_cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await proc.communicate()
    
    if proc.returncode != 0:
        raise Exception(f"Command failed: {stderr.decode().strip()}")
    return stdout.decode()

async def _get_current_cwd() -> str:
    """内部辅助：获取当前配置的工作目录"""
    settings = await load_settings()
    cwd = settings.get("CLISettings", {}).get("cc_path")
    if not cwd:
        raise ValueError("No workspace directory specified in settings (CLISettings.cc_path).")
    return cwd

async def docker_sandbox_async(command: str) -> str | AsyncIterator[str]:
    """
    在持久化 Docker 沙盒中执行命令
    """
    settings = await load_settings()
    CLISettings = settings.get("CLISettings", {})
    cwd = CLISettings.get("cc_path")
    if not cwd:
        return "Error: No workspace directory specified in settings."
    dsSettings = settings.get("dsSettings", {})
    
    image_name = "docker/sandbox-templates:latest"
    
    if not cwd or not Path(cwd).is_dir():
        return f"Error: Invalid workspace directory: {cwd}"
    
    try:
        container_name = await get_or_create_docker_sandbox(cwd, image_name)
    except Exception as e:
        return f"Docker Sandbox Initialization Error: {str(e)}"

    async def _stream() -> AsyncIterator[str]:
        exec_cmd = [
            "docker", "exec",
            "-i",
            container_name,
            "sh", "-c",
            f"cd /workspace && {command}"
        ]
        
        output_yielded = False
        
        try:
            process = await asyncio.create_subprocess_exec(
                *exec_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            
            async for line in _merge_streams(
                read_stream(process.stdout, is_error=False),
                read_stream(process.stderr, is_error=True),
            ):
                yield line
                output_yielded = True
            
            await process.wait()
            
            if process.returncode != 0:
                yield f"[EXIT CODE] {process.returncode}"
            elif process.returncode == 0 and not output_yielded:
                yield "[SUCCESS] 命令已成功执行未报错"
                
        except Exception as e:
            yield f"[ERROR] 执行失败: {str(e)}"
    
    return _stream()

# ==================== 1. 精确字符串替换工具 (edit_file_patch) ====================

async def edit_file_patch_tool(path: str, old_string: str, new_string: str) -> str:
    """
    [工具] 精确字符串替换 - Claude Code 经典功能
    查找特定代码块并替换，保留文件其余部分和格式
    
    特性：
    - 精确匹配 old_string（去除行尾空格后进行匹配）
    - 只替换第一个匹配项
    - 如果匹配失败，返回详细错误信息帮助定位
    """
    try:
        real_cwd = await _get_current_cwd()
        container_name = await get_or_create_docker_sandbox(real_cwd)
        
        # 读取文件内容
        content = await _exec_docker_cmd_simple(real_cwd, ["cat", path])
        
        # 规范化行尾空格用于匹配（但保留原文件格式）
        normalized_content = "\n".join(line.rstrip() for line in content.split("\n"))
        normalized_old = "\n".join(line.rstrip() for line in old_string.split("\n"))
        
        if normalized_old not in normalized_content:
            # 提供诊断信息
            lines = content.split("\n")
            first_line = old_string.split("\n")[0] if "\n" in old_string else old_string
            
            # 尝试模糊查找第一行
            similar_lines = [f"Line {i+1}: {line[:80]}" for i, line in enumerate(lines) 
                           if first_line.strip() in line]
            
            error_msg = f"[Error] Old string not found in file '{path}'.\n"
            if similar_lines:
                error_msg += f"\nFound similar lines containing '{first_line[:30]}':\n" + "\n".join(similar_lines[:5])
            else:
                error_msg += f"\nFile has {len(lines)} lines. First line of your search: '{first_line[:50]}'"
            return error_msg
        
        # 执行替换（使用原始内容）
        new_content = content.replace(old_string, new_string, 1)
        
        # 写回文件
        with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8') as tmp:
            tmp.write(new_content)
            tmp_path = tmp.name
        
        dest_path = f"{container_name}:/workspace/{path}"
        
        cp_proc = await asyncio.create_subprocess_exec(
            "docker", "cp", tmp_path, dest_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        _, stderr = await cp_proc.communicate()
        
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        
        if cp_proc.returncode != 0:
            return f"[Error] Patch failed: {stderr.decode()}"
        
        # 添加行数统计信息
        old_lines = old_string.count('\n') + 1
        new_lines = new_string.count('\n') + 1
        return f"[Success] Patched '{path}' ({old_lines} lines -> {new_lines} lines)"
        
    except Exception as e:
        return f"[Error] Patch failed: {str(e)}"

# ==================== 2. Glob 文件匹配工具 (glob_files) ====================

async def glob_files_tool(pattern: str, exclude: str = "**/node_modules/**,**/.git/**,**/__pycache__/**") -> str:
    """
    [工具] 真正的 Glob 模式匹配（递归查找）
    支持 **/*.py 等递归模式，弥补了 list_files 只能列单层目录的不足
    
    参数:
        pattern: glob 模式，如 "**/*.py", "src/**/*.ts", "*.md"
        exclude: 排除模式，逗号分隔（默认排除 node_modules, .git, __pycache__）
    """
    try:
        real_cwd = await _get_current_cwd()
        
        # 在容器内使用 Python 的 glob 模块（最准确）
        exclude_list = [e.strip() for e in exclude.split(",") if e.strip()]
        
        python_script = f'''
import glob
import os
import json

files = glob.glob("/workspace/{pattern}", recursive=True)
exclude_patterns = {exclude_list}

filtered = []
for f in files:
    if not os.path.isfile(f):
        continue
    rel_path = f.replace("/workspace/", "")
    # 检查排除模式
    should_exclude = False
    for ex in exclude_patterns:
        if glob.fnmatch.fnmatch(rel_path, ex) or glob.fnmatch.fnmatch(f, ex):
            should_exclude = True
            break
    if not should_exclude:
        filtered.append(rel_path)

print(json.dumps(filtered, indent=2))
'''
        
        output = await _exec_docker_cmd_simple(real_cwd, ["python3", "-c", python_script])
        
        try:
            files = json.loads(output)
            if not files:
                return "[Result] No files found matching the pattern."
            
            # 格式化输出，带序号和文件类型标识
            result_lines = [f"[{len(files)} files matched]"]
            for i, f in enumerate(files[:50], 1):  # 限制显示前50个
                icon = "📄" if "." in f else "📁"
                if f.endswith(".py"): icon = "🐍"
                elif f.endswith(".js") or f.endswith(".ts"): icon = "📜"
                elif f.endswith(".md"): icon = "📝"
                elif f.endswith(".json"): icon = "⚙️"
                result_lines.append(f"{icon} {f}")
            
            if len(files) > 50:
                result_lines.append(f"\n... and {len(files) - 50} more files")
            
            return "\n".join(result_lines)
            
        except json.JSONDecodeError:
            return f"[Result] {output}"
            
    except Exception as e:
        return f"[Error] Glob failed: {str(e)}"

# ==================== 3. 任务管理工具 (todo_write) ====================

async def todo_write_tool(action: str, id: str = None, content: str = None, priority: str = "medium", status: str = None) -> str:
    """
    [工具] 完整的任务管理系统
    持久化存储在 .party/ai_todos.json，支持优先级和状态跟踪
    
    操作:
        create: 创建新任务 (需要 content, 可选 priority)
        update: 更新任务 (需要 id, 可选 content/priority/status)
        delete: 删除任务 (需要 id)
        list: 列出所有任务
        toggle: 切换任务完成状态 (需要 id)
        
    优先级: high, medium, low
    状态: pending, in_progress, done, cancelled
    """
    try:
        real_cwd = await _get_current_cwd()
        container_name = await get_or_create_docker_sandbox(real_cwd)
        
        todo_dir = "/workspace/.party"
        todo_file = f"{todo_dir}/ai_todos.json"
        
        # 读取现有 todos
        try:
            content_data = await _exec_docker_cmd_simple(real_cwd, ["cat", todo_file])
            todos = json.loads(content_data)
            if not isinstance(todos, list):
                todos = []
        except Exception:
            todos = []
        
        # 执行操作
        if action == "create":
            if not content:
                return "[Error] 'content' is required for create action"
            
            new_todo = {
                "id": id or str(uuid.uuid4())[:8],
                "content": content,
                "priority": priority if priority in ["high", "medium", "low"] else "medium",
                "status": "pending",
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "completed_at": None
            }
            todos.append(new_todo)
            
            # 写回文件
            json_str = json.dumps(todos, indent=2, ensure_ascii=False)
            with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8') as tmp:
                tmp.write(json_str)
                tmp_path = tmp.name
            
            # 确保目录存在
            await _exec_docker_cmd_simple(real_cwd, ["mkdir", "-p", todo_dir])
            
            dest_path = f"{container_name}:{todo_file}"
            cp_proc = await asyncio.create_subprocess_exec(
                "docker", "cp", tmp_path, dest_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await cp_proc.communicate()
            os.unlink(tmp_path)
            
            return f"[Success] Created todo [{new_todo['id']}]: {content}"
            
        elif action == "list":
            if not todos:
                return "[Result] No todos found. Create one with action='create'"
            
            # 按优先级和状态排序
            priority_order = {"high": 0, "medium": 1, "low": 2}
            sorted_todos = sorted(todos, key=lambda x: (priority_order.get(x.get('priority', 'medium'), 1), 
                                                        x.get('status', 'pending') != 'pending'))
            
            lines = ["📋 Task List:", "─" * 50]
            for t in sorted_todos:
                status_icon = {"pending": "⏳", "in_progress": "🔄", "done": "✅", "cancelled": "❌"}.get(t.get('status'), "⏳")
                priority_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(t.get('priority'), "🟡")
                lines.append(f"{status_icon} [{t['id']}] {t['content'][:40]} {priority_icon}")
                if len(t['content']) > 40:
                    lines.append(f"    ...{t['content'][40:]}")
            
            lines.append("─" * 50)
            lines.append(f"Total: {len(todos)} tasks ({sum(1 for t in todos if t.get('status') != 'done')} pending)")
            return "\n".join(lines)
            
        elif action == "update":
            if not id:
                return "[Error] 'id' is required for update action"
            
            found = False
            for todo in todos:
                if todo["id"] == id:
                    if content:
                        todo["content"] = content
                    if priority and priority in ["high", "medium", "low"]:
                        todo["priority"] = priority
                    if status and status in ["pending", "in_progress", "done", "cancelled"]:
                        todo["status"] = status
                        if status == "done" and not todo.get("completed_at"):
                            todo["completed_at"] = datetime.now().isoformat()
                    todo["updated_at"] = datetime.now().isoformat()
                    found = True
                    break
            
            if not found:
                return f"[Error] Todo with id '{id}' not found. Use action='list' to see all ids."
            
            # 写回
            json_str = json.dumps(todos, indent=2, ensure_ascii=False)
            with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8') as tmp:
                tmp.write(json_str)
                tmp_path = tmp.name
            
            dest_path = f"{container_name}:{todo_file}"
            await asyncio.create_subprocess_exec("docker", "cp", tmp_path, dest_path,
                                                stdout=asyncio.subprocess.PIPE, 
                                                stderr=asyncio.subprocess.PIPE)
            os.unlink(tmp_path)
            
            return f"[Success] Updated todo [{id}]"
            
        elif action == "delete":
            if not id:
                return "[Error] 'id' is required for delete action"
            
            original_len = len(todos)
            todos = [t for t in todos if t["id"] != id]
            
            if len(todos) == original_len:
                return f"[Error] Todo with id '{id}' not found."
            
            # 写回
            json_str = json.dumps(todos, indent=2, ensure_ascii=False)
            with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8') as tmp:
                tmp.write(json_str)
                tmp_path = tmp.name
            
            dest_path = f"{container_name}:{todo_file}"
            await asyncio.create_subprocess_exec("docker", "cp", tmp_path, dest_path,
                                                stdout=asyncio.subprocess.PIPE,
                                                stderr=asyncio.subprocess.PIPE)
            os.unlink(tmp_path)
            
            return f"[Success] Deleted todo [{id}]"
            
        elif action == "toggle":
            if not id:
                return "[Error] 'id' is required for toggle action"
            
            for todo in todos:
                if todo["id"] == id:
                    if todo.get("status") == "done":
                        todo["status"] = "pending"
                        todo["completed_at"] = None
                        msg = "marked as pending"
                    else:
                        todo["status"] = "done"
                        todo["completed_at"] = datetime.now().isoformat()
                        msg = "completed"
                    
                    todo["updated_at"] = datetime.now().isoformat()
                    
                    # 写回
                    json_str = json.dumps(todos, indent=2, ensure_ascii=False)
                    with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8') as tmp:
                        tmp.write(json_str)
                        tmp_path = tmp.name
                    
                    dest_path = f"{container_name}:{todo_file}"
                    await asyncio.create_subprocess_exec("docker", "cp", tmp_path, dest_path,
                                                        stdout=asyncio.subprocess.PIPE,
                                                        stderr=asyncio.subprocess.PIPE)
                    os.unlink(tmp_path)
                    
                    return f"[Success] Todo [{id}] {msg} ✅"
            
            return f"[Error] Todo with id '{id}' not found."
            
        else:
            return f"[Error] Unknown action: {action}. Use: create, list, update, delete, toggle"
            
    except Exception as e:
        return f"[Error] Todo operation failed: {str(e)}"

# ==================== 工具注册与权限管理 ====================

TOOLS_REGISTRY = {
    # --- 只读工具 ---
    "list_files": {
        "type": "function",
        "function": {
            "name": "list_files_tool",
            "description": "List files and directories in the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "The directory path (default: .)"},
                    "show_all": {"type": "boolean", "description": "Show hidden files (default: false)"}
                }
            }
        }
    },
    "read_file": {
        "type": "function",
        "function": {
            "name": "read_file_tool",
            "description": "Read the contents of a file with line numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "The path to the file"}
                },
                "required": ["path"]
            }
        }
    },
    "search_files": {
        "type": "function",
        "function": {
            "name": "search_files_tool",
            "description": "Search for a text pattern recursively in files using grep.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "The regex or text to search for"},
                    "path": {"type": "string", "description": "Directory to search in (default: .)"}
                },
                "required": ["pattern"]
            }
        }
    },
    "glob_files": {
        "type": "function",
        "function": {
            "name": "glob_files_tool",
            "description": "Find files using glob patterns (e.g., '**/*.py' for all Python files recursively). Much more powerful than list_files for finding specific file types across the project.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string", 
                        "description": "Glob pattern like '**/*.py', 'src/**/*.ts', '*.md', 'test_*.py'"
                    },
                    "exclude": {
                        "type": "string",
                        "description": "Comma-separated exclusion patterns (default: '**/node_modules/**,**/.git/**')"
                    }
                },
                "required": ["pattern"]
            }
        }
    },
    
    # --- 编辑工具 ---
    "edit_file": {
        "type": "function",
        "function": {
            "name": "edit_file_tool",
            "description": "Create or Overwrite a file with new content. For editing, read the file first, then provide the FULL new content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "The file path"},
                    "content": {"type": "string", "description": "The full content to write to the file"}
                },
                "required": ["path", "content"]
            }
        }
    },
    "edit_file_patch": {
        "type": "function",
        "function": {
            "name": "edit_file_patch_tool",
            "description": "Precise string replacement - the classic Claude Code feature. Finds a specific code block (old_string) and replaces it with new_string, preserving the rest of the file. Safer than edit_file for modifications. old_string must match exactly (except trailing whitespace).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to edit"
                    },
                    "old_string": {
                        "type": "string",
                        "description": "The exact code block to replace (can be multiple lines). Must match the file content precisely."
                    },
                    "new_string": {
                        "type": "string",
                        "description": "The new code block to insert in place of old_string"
                    }
                },
                "required": ["path", "old_string", "new_string"]
            }
        }
    },
    
    # --- 任务管理工具 ---
    "todo_write": {
        "type": "function",
        "function": {
            "name": "todo_write_tool",
            "description": "Task management system with persistent storage in .party/ai_todos.json. CRUD operations for project tasks with priorities and status tracking. Actions: create (needs content), list, update (needs id), delete (needs id), toggle (toggle done status).",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["create", "list", "update", "delete", "toggle"],
                        "description": "Operation to perform"
                    },
                    "id": {
                        "type": "string",
                        "description": "Task ID (required for update/delete/toggle, optional for create)"
                    },
                    "content": {
                        "type": "string",
                        "description": "Task description (required for create, optional for update)"
                    },
                    "priority": {
                        "type": "string",
                        "enum": ["high", "medium", "low"],
                        "description": "Task priority (default: medium)"
                    },
                    "status": {
                        "type": "string",
                        "enum": ["pending", "in_progress", "done", "cancelled"],
                        "description": "Task status (for update action)"
                    }
                },
                "required": ["action"]
            }
        }
    },
    
    # --- 全权限工具 (Bash) ---
    "bash": {
        "type": "function",
        "function": {
            "name": "docker_sandbox_async", 
            "description": "Execute a bash command in the terminal. Use this for running scripts, installing packages, or git operations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The bash command"}
                },
                "required": ["command"]
            }
        }
    }
}

def get_tools_for_mode(mode: str) -> list:
    """
    根据权限模式返回工具定义列表
    
    权限矩阵:
    - default (Default Permission Mode): 只读工具
    - auto-approve (Accept Edits): 只读 + 文件编辑 + 任务管理  
    - yolo (Bypass Permissions): 全部工具（包括 bash）
    """
    
    # 基础只读集
    read_only_tools = [
        TOOLS_REGISTRY["list_files"],
        TOOLS_REGISTRY["read_file"],
        TOOLS_REGISTRY["search_files"],
        TOOLS_REGISTRY["glob_files"]  # 新增：递归文件查找（只读）
    ]
    
    # 编辑集 (文件修改)
    edit_tools = [
        TOOLS_REGISTRY["edit_file"],
        TOOLS_REGISTRY["edit_file_patch"]  # 新增：精确字符串替换（比全量覆盖更安全）
    ]
    
    # 任务管理集 (元数据操作，理论上安全，但涉及文件写入)
    todo_tools = [
        TOOLS_REGISTRY["todo_write"]  # 新增：任务管理系统
    ]
    
    # 终端集 (危险操作)
    terminal_tools = [
        TOOLS_REGISTRY["bash"]
    ]
    
    if mode == "default":
        # 默认模式：只能浏览和搜索
        return read_only_tools
        
    elif mode == "auto-approve": 
        # 自动批准模式：可以读写文件和管理任务，但不能执行任意 bash
        return read_only_tools + edit_tools + todo_tools
        
    elif mode == "yolo":
        # 完全授权模式：所有工具
        return read_only_tools + edit_tools + todo_tools + terminal_tools
    
    else:
        # 未知模式，返回最安全选项
        return read_only_tools

# ==================== 其他原有工具函数 ====================

async def read_file_tool(path: str) -> str:
    """[工具] 读取文件内容，带有行号"""
    try:
        real_cwd = await _get_current_cwd()
        cmd = ["cat", "-n", path] 
        output = await _exec_docker_cmd_simple(real_cwd, cmd)
        return output
    except Exception as e:
        return f"[Error] Could not read file: {str(e)}"

async def edit_file_tool(path: str, content: str) -> str:
    """[工具] 覆盖写入文件"""
    try:
        real_cwd = await _get_current_cwd()
        container_name = await get_or_create_docker_sandbox(real_cwd)
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8') as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        
        dir_name = os.path.dirname(path)
        if dir_name:
            await _exec_docker_cmd_simple(real_cwd, ["mkdir", "-p", dir_name])

        dest_path = f"{container_name}:/workspace/{path}"
        
        cp_proc = await asyncio.create_subprocess_exec(
            "docker", "cp", tmp_path, dest_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        _, stderr = await cp_proc.communicate()
        
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        
        if cp_proc.returncode != 0:
            return f"[Error] Save failed: {stderr.decode()}"
            
        return f"[Success] File '{path}' saved successfully."
    except Exception as e:
        return f"[Error] Edit tool failed: {str(e)}"

async def search_files_tool(pattern: str, path: str = ".") -> str:
    """[工具] 使用 grep 搜索文件内容"""
    try:
        real_cwd = await _get_current_cwd()
        cmd = ["grep", "-rn", pattern, path]
        output = await _exec_docker_cmd_simple(real_cwd, cmd)
        return output if output else "[Result] No matches found."
    except Exception as e:
        return f"[Error] Search failed: {str(e)}"

async def list_files_tool(path: str = ".", show_all: bool = False) -> str:
    """[工具] 列出目录下的文件
    参数:
        show_all: 是否显示隐藏文件（默认False）
    """
    try:
        real_cwd = await _get_current_cwd()
        
        # 先检查是否有隐藏文件
        all_files = await _exec_docker_cmd_simple(real_cwd, ["ls", "-A", path])
        has_hidden = any(f.startswith('.') for f in all_files.split('\n') if f)
        
        if show_all:
            cmd = ["ls", "-laF", path]
            output = await _exec_docker_cmd_simple(real_cwd, cmd)
            if not output:
                if has_hidden:
                    return "[Result] 当前目录没有可见文件，但包含隐藏项目（如 .party, .git 等）。如需查看请使用 show_all=true"
                else:
                    return "[Result] Directory is empty."
            return output
        else:
            # 默认不显示隐藏文件
            cmd = ["ls", "-F", path]
            output = await _exec_docker_cmd_simple(real_cwd, cmd)
            
            if not output:
                if has_hidden:
                    return "[Result] 当前目录没有可见文件，但包含隐藏项目（如 .party, .git 等）。如需查看请使用 show_all=true"
                else:
                    return "[Result] Directory is empty."
            return output
            
    except Exception as e:
        return f"[Error] {str(e)}"



class LocalEnvConfig:
    """本地环境配置管理"""
    def __init__(self):
        self.permission_mode = "default"
        self.workspace = ""
    
    @classmethod
    async def from_settings(cls) -> "LocalEnvConfig":
        """从设置加载配置"""
        config = cls()
        settings = await load_settings()
        cli_settings = settings.get("CLISettings", {})
        local_settings = settings.get("localEnvSettings", {})
        
        config.workspace = cli_settings.get("cc_path", "")
        config.permission_mode = local_settings.get("permissionMode", "default")
        return config

def get_safe_workspace_path(cwd: str, sub_path: str = "") -> Path:
    """
    安全的路径解析：确保所有操作都在工作空间内
    防止路径遍历攻击 (Path Traversal)
    """
    base = Path(cwd).resolve()
    if sub_path:
        target = (base / sub_path).resolve()
        try:
            target.relative_to(base)
            return target
        except ValueError:
            raise PermissionError(f"Path '{sub_path}' is outside of workspace '{cwd}'")
    return base

# ==================== 流处理工具（复用）====================

async def read_stream_local(stream, *, is_error: bool = False):
    """读取流并添加错误前缀"""
    if stream is None:
        return
    async for line in stream:
        prefix = "[ERROR] " if is_error else ""
        yield f"{prefix}{line.decode('utf-8', errors='replace').rstrip()}"

async def _merge_streams_local(*streams):
    """合并多个异步流"""
    streams = [s.__aiter__() for s in streams]
    while streams:
        for stream in list(streams):
            try:
                item = await stream.__anext__()
                yield item
            except StopAsyncIteration:
                streams.remove(stream)

async def _get_current_cwd_local() -> str:
    """获取当前配置的工作目录"""
    settings = await load_settings()
    cwd = settings.get("CLISettings", {}).get("cc_path")
    if not cwd:
        raise ValueError("No workspace directory specified in settings (CLISettings.cc_path).")
    if not Path(cwd).is_dir():
        raise ValueError(f"Workspace directory does not exist: {cwd}")
    return cwd

# ==================== 纯跨平台本地环境工具 ====================

async def _get_current_cwd_local() -> str:
    """获取当前配置的工作目录（跨平台）"""
    settings = await load_settings()
    cwd = settings.get("CLISettings", {}).get("cc_path")
    if not cwd:
        raise ValueError("No workspace directory specified in settings (CLISettings.cc_path).")
    
    # 跨平台路径处理
    cwd_path = Path(cwd).resolve()
    if not cwd_path.exists():
        raise ValueError(f"Workspace directory does not exist: {cwd}")
    return str(cwd_path)

def get_safe_workspace_path(cwd: str, sub_path: str = "") -> Path:
    """安全的路径解析（跨平台）"""
    base = Path(cwd).resolve()
    if sub_path:
        # 标准化路径分隔符（Windows 使用 \，Unix 使用 /）
        target = (base / sub_path).resolve()
        try:
            # 确保目标路径在工作空间内
            target.relative_to(base)
            return target
        except ValueError:
            raise PermissionError(f"Path '{sub_path}' is outside of workspace '{cwd}'")
    return base

async def read_todos_local(cwd: str) -> list:
    """读取本地待办事项（跨平台，不依赖外部命令）"""
    todo_file = Path(cwd) / ".party" / "ai_todos.json"
    if not todo_file.exists():
        return []
    
    try:
        async with aiofiles.open(todo_file, 'r', encoding='utf-8') as f:
            content = await f.read()
            if not content.strip():
                return []
            return json.loads(content)
    except (json.JSONDecodeError, FileNotFoundError):
        return []
    except Exception as e:
        print(f"[Todo Loader] Error reading todos: {e}")
        return []

# 1. 跨平台搜索工具（不依赖 grep/rg）
async def search_files_tool_local(pattern: str, path: str = ".") -> str:
    """
    [本地环境-跨平台] 递归搜索文件内容
    使用 Python 原生实现，不依赖系统 grep 命令
    """
    try:
        cwd = await _get_current_cwd_local()
        target_dir = get_safe_workspace_path(cwd, path)
        
        matches = []
        compiled_pattern = re.compile(pattern)
        
        # 递归遍历目录
        for root, dirs, files in os.walk(target_dir):
            # 跳过隐藏目录和常见依赖目录（跨平台）
            dirs[:] = [d for d in dirs if not d.startswith('.') and 
                      d not in ['__pycache__', 'node_modules', 'venv', '.git', 'dist', 'build']]
            
            for file in files:
                if file.startswith('.'):
                    continue
                    
                file_path = Path(root) / file
                
                # 只搜索文本文件，跳过二进制文件
                try:
                    # 异步读取文件
                    async with aiofiles.open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = await f.read()
                        
                    lines = content.splitlines()
                    for i, line in enumerate(lines, 1):
                        if compiled_pattern.search(line):
                            rel_path = file_path.relative_to(target_dir)
                            matches.append(f"{rel_path}:{i}:{line.strip()}")
                            
                            # 限制结果数量，避免返回过多
                            if len(matches) >= 100:
                                break
                    if len(matches) >= 100:
                        break
                        
                except (IOError, OSError):
                    continue
            
            if len(matches) >= 100:
                break
        
        if not matches:
            return "[Result] No matches found."
        
        result = "\n".join(matches[:100])
        if len(matches) >= 100:
            result += "\n[Note] More than 100 matches found, showing first 100."
        return result
        
    except Exception as e:
        return f"[Error] Search failed: {str(e)}"

# 2. 跨平台 Bash 工具（自动适配操作系统）
async def bash_tool_local(command: str) -> str | AsyncIterator[str]:
    """
    [本地环境-跨平台] 执行命令
    Windows 使用 cmd，macOS/Linux 使用 bash/sh
    """
    settings = await load_settings()
    cwd = settings.get("CLISettings", {}).get("cc_path")
    local_settings = settings.get("localEnvSettings", {})
    permission_mode = local_settings.get("permissionMode", "default")
    
    if not cwd:
        return "Error: No workspace directory specified in settings."
    
    cwd_path = Path(cwd)
    if not cwd_path.exists():
        return f"Error: Invalid workspace directory: {cwd}"
    
    # 安全限制（跨平台）
    dangerous_patterns = [
        r'rm\s+-rf\s+/[^ ]*$',  # rm -rf /something
        r'mkfs\.',               # 格式化
        r'dd\s+if=',             # dd 操作
        r'>\s*/dev/sda',         # 写入磁盘
        r'format\s+[a-zA-Z]:',   # Windows 格式化
        r'del\s+/[fq]',          # Windows 强制删除
    ]
    
    if permission_mode != "yolo":
        for pattern in dangerous_patterns:
            if re.search(pattern, command, re.IGNORECASE):
                return f"[Error] Dangerous command blocked in '{permission_mode}' mode: {command[:50]}..."
    
    # 根据操作系统选择 shell
    system = platform.system()
    
    if system == "Windows":
        # Windows: 使用 cmd 或 PowerShell
        # 检测是否使用 PowerShell 命令
        if any(cmd in command.lower() for cmd in ['get-', 'set-', 'write-', '|', 'select-object', 'where-object']):
            # PowerShell 命令
            executable = "powershell.exe"
            args = ["-Command", command]
        else:
            # CMD 命令
            executable = "cmd.exe"
            args = ["/c", command]
    else:
        # macOS/Linux: 使用 sh 或 bash
        shell = os.environ.get('SHELL', '/bin/bash')
        executable = shell
        args = ["-c", command]
    
    async def _stream() -> AsyncIterator[str]:
        try:
            process = await asyncio.create_subprocess_exec(
                executable,
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(cwd_path),
                env=os.environ.copy()
            )
            
            output_yielded = False
            
            # 使用通用的 read_stream 函数
            async for line in _merge_streams(
                read_stream(process.stdout, is_error=False),
                read_stream(process.stderr, is_error=True),
            ):
                yield line
                output_yielded = True
            
            await process.wait()
            
            if process.returncode != 0:
                yield f"[EXIT CODE] {process.returncode}"
            elif process.returncode == 0 and not output_yielded:
                yield "[SUCCESS] Command executed successfully (no output)"
                
        except Exception as e:
            yield f"[ERROR] Execution failed: {str(e)}"
    
    return _stream()

# 3. 跨平台文件列表（修复可执行文件检测）
async def list_files_tool_local(path: str = ".", show_all: bool = False) -> str:
    """
    [本地环境-跨平台] 列出目录内容
    适配 Windows 和 Unix 的可执行文件检测
    """
    try:
        cwd = await _get_current_cwd_local()
        target_dir = get_safe_workspace_path(cwd, path)
        
        entries = []
        
        # 使用 Path 迭代（跨平台）
        for entry in target_dir.iterdir():
            # 隐藏文件处理
            if not show_all and entry.name.startswith('.'):
                continue
            
            suffix = ""
            try:
                if entry.is_dir():
                    suffix = "/"
                elif entry.is_symlink():
                    suffix = "@"
                elif entry.is_file():
                    # 跨平台可执行文件检测
                    if _is_executable(entry):
                        suffix = "*"
            except OSError:
                # 某些文件可能无法访问（权限问题）
                continue
            
            entries.append(f"{entry.name}{suffix}")
        
        if not entries:
            # 检查是否有隐藏文件
            try:
                has_hidden = any(e.name.startswith('.') for e in target_dir.iterdir() if e.is_file() or e.is_dir())
                if has_hidden and not show_all:
                    return "[Result] 当前目录没有可见文件，但包含隐藏项目。如需查看请使用 show_all=true"
            except:
                pass
            return "[Result] Directory is empty."
        
        # 排序：目录在前，文件在后，按字母顺序（不区分大小写）
        entries.sort(key=lambda x: (
            not x.endswith('/'),  # 目录在前
            x.lower().rstrip('*@/')  # 不区分大小写，去掉标记符后排序
        ))
        
        return "\n".join(entries)
            
    except Exception as e:
        return f"[Error] {str(e)}"

def _is_executable(file_path: Path) -> bool:
    """
    跨平台可执行文件检测
    """
    try:
        if platform.system() == "Windows":
            # Windows: 检查扩展名
            executable_extensions = {'.exe', '.bat', '.cmd', '.ps1', '.py', '.sh', '.com'}
            return file_path.suffix.lower() in executable_extensions
        else:
            # Unix/Linux/macOS: 使用 os.access
            return os.access(file_path, os.X_OK)
    except:
        return False

# 4. 读取文件（已经是跨平台的，只需确保编码处理）
async def read_file_tool_local(path: str) -> str:
    """[本地环境] 读取文件内容（跨平台）"""
    try:
        cwd = await _get_current_cwd_local()
        safe_path = get_safe_workspace_path(cwd, path)
        
        lines = []
        async with aiofiles.open(safe_path, 'r', encoding='utf-8', errors='replace') as f:
            content = await f.read()
            
        for i, line in enumerate(content.splitlines(), 1):
            lines.append(f"{i:6}\t{line.rstrip()}")
        
        return "\n".join(lines) if lines else "[Result] File is empty."
    except Exception as e:
        return f"[Error] Could not read file: {str(e)}"

# 5. 写入文件（已经是跨平台的）
async def edit_file_tool_local(path: str, content: str) -> str:
    """[本地环境-跨平台] 写入文件"""
    try:
        cwd = await _get_current_cwd_local()
        safe_path = get_safe_workspace_path(cwd, path)
        
        # 确保父目录存在（跨平台）
        await aiofiles.os.makedirs(safe_path.parent, exist_ok=True)
        
        async with aiofiles.open(safe_path, 'w', encoding='utf-8') as f:
            await f.write(content)
            
        return f"[Success] File '{path}' saved successfully."
    except Exception as e:
        return f"[Error] Edit tool failed: {str(e)}"

# 6. Glob 工具（已经是跨平台的，使用标准库 glob）
async def glob_files_tool_local(pattern: str, exclude: str = "**/node_modules/**,**/.git/**,**/__pycache__/**") -> str:
    """[本地环境-跨平台] Glob 文件匹配"""
    try:
        cwd = await _get_current_cwd_local()
        base_path = Path(cwd)
        
        exclude_list = [e.strip() for e in exclude.split(",") if e.strip()]
        
        # 使用标准库 glob（已经是跨平台的）
        full_pattern = str(base_path / pattern)
        files = std_glob.glob(full_pattern, recursive=True)
        
        filtered = []
        for f in files:
            p = Path(f)
            if not p.is_file():
                continue
            
            try:
                rel_path = str(p.relative_to(base_path))
            except ValueError:
                continue
            
            # 检查排除模式
            should_exclude = False
            for ex in exclude_list:
                if fnmatch.fnmatch(rel_path, ex) or fnmatch.fnmatch(f, ex):
                    should_exclude = True
                    break
            
            if not should_exclude:
                filtered.append(rel_path)
        
        if not filtered:
            return "[Result] No files found matching the pattern."
        
        # 格式化输出
        result_lines = [f"[{len(filtered)} files matched]"]
        for i, f in enumerate(filtered[:50], 1):
            icon = "📄"
            if f.endswith(".py"): icon = "🐍"
            elif f.endswith(".js") or f.endswith(".ts"): icon = "📜"
            elif f.endswith(".md"): icon = "📝"
            elif f.endswith(".json"): icon = "⚙️"
            result_lines.append(f"{icon} {f}")
        
        if len(filtered) > 50:
            result_lines.append(f"\n... and {len(filtered) - 50} more files")
        
        return "\n".join(result_lines)
        
    except Exception as e:
        return f"[Error] Glob failed: {str(e)}"

# 7. 精确替换工具（已经是跨平台的）
async def edit_file_patch_tool_local(path: str, old_string: str, new_string: str) -> str:
    """[本地环境-跨平台] 精确字符串替换"""
    try:
        cwd = await _get_current_cwd_local()
        safe_path = get_safe_workspace_path(cwd, path)
        
        async with aiofiles.open(safe_path, 'r', encoding='utf-8', errors='replace') as f:
            content = await f.read()
        
        # 规范化行尾空格用于匹配
        normalized_content = "\n".join(line.rstrip() for line in content.split("\n"))
        normalized_old = "\n".join(line.rstrip() for line in old_string.split("\n"))
        
        if normalized_old not in normalized_content:
            lines = content.split("\n")
            first_line = old_string.split("\n")[0] if "\n" in old_string else old_string
            
            similar_lines = [f"Line {i+1}: {line[:80]}" for i, line in enumerate(lines) 
                           if first_line.strip() in line]
            
            error_msg = f"[Error] Old string not found in file '{path}'.\n"
            if similar_lines:
                error_msg += f"\nFound similar lines containing '{first_line[:30]}':\n" + "\n".join(similar_lines[:5])
            else:
                error_msg += f"\nFile has {len(lines)} lines. First line of your search: '{first_line[:50]}'"
            return error_msg
        
        new_content = content.replace(old_string, new_string, 1)
        
        async with aiofiles.open(safe_path, 'w', encoding='utf-8') as f:
            await f.write(new_content)
        
        old_lines = old_string.count('\n') + 1
        new_lines = new_string.count('\n') + 1
        return f"[Success] Patched '{path}' ({old_lines} lines -> {new_lines} lines)"
        
    except Exception as e:
        return f"[Error] Patch failed: {str(e)}"

# 8. 待办事项工具（已经是跨平台的）
async def todo_write_tool_local(action: str, id: str = None, content: str = None, priority: str = "medium", status: str = None) -> str:
    """[本地环境-跨平台] 任务管理"""
    try:
        cwd = await _get_current_cwd_local()
        party_dir = Path(cwd) / ".party"
        todo_file = party_dir / "ai_todos.json"
        
        # 创建目录（跨平台）
        await aiofiles.os.makedirs(party_dir, exist_ok=True)
        
        # 读取
        todos = await read_todos_local(cwd)
        if not isinstance(todos, list):
            todos = []
        
        # 处理各种操作（与之前相同，使用纯 Python 文件操作）
        if action == "create":
            if not content:
                return "[Error] 'content' is required for create action"
            
            new_todo = {
                "id": id or str(uuid.uuid4())[:8],
                "content": content,
                "priority": priority if priority in ["high", "medium", "low"] else "medium",
                "status": "pending",
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "completed_at": None
            }
            todos.append(new_todo)
            
            async with aiofiles.open(todo_file, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(todos, indent=2, ensure_ascii=False))
            
            return f"[Success] Created todo [{new_todo['id']}]: {content}"
            
        elif action == "list":
            if not todos:
                return "[Result] No todos found. Create one with action='create'"
            
            priority_order = {"high": 0, "medium": 1, "low": 2}
            sorted_todos = sorted(todos, key=lambda x: (priority_order.get(x.get('priority', 'medium'), 1), 
                                                        x.get('status', 'pending') != 'pending'))
            
            lines = ["📋 Task List:", "─" * 50]
            for t in sorted_todos:
                status_icon = {"pending": "⏳", "in_progress": "🔄", "done": "✅", "cancelled": "❌"}.get(t.get('status'), "⏳")
                priority_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(t.get('priority'), "🟡")
                lines.append(f"{status_icon} [{t['id']}] {t['content'][:40]} {priority_icon}")
                if len(t['content']) > 40:
                    lines.append(f"    ...{t['content'][40:]}")
            
            lines.append("─" * 50)
            lines.append(f"Total: {len(todos)} tasks ({sum(1 for t in todos if t.get('status') != 'done')} pending)")
            return "\n".join(lines)
            
        elif action == "update":
            if not id:
                return "[Error] 'id' is required for update action"
            
            found = False
            for todo in todos:
                if todo["id"] == id:
                    if content:
                        todo["content"] = content
                    if priority and priority in ["high", "medium", "low"]:
                        todo["priority"] = priority
                    if status and status in ["pending", "in_progress", "done", "cancelled"]:
                        todo["status"] = status
                        if status == "done":
                            todo["completed_at"] = datetime.now().isoformat()
                    todo["updated_at"] = datetime.now().isoformat()
                    found = True
                    break
            
            if not found:
                return f"[Error] Todo with id '{id}' not found."
            
            async with aiofiles.open(todo_file, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(todos, indent=2, ensure_ascii=False))
            return f"[Success] Updated todo [{id}]"
            
        elif action == "delete":
            if not id:
                return "[Error] 'id' is required for delete action"
            
            original_len = len(todos)
            todos = [t for t in todos if t["id"] != id]
            
            if len(todos) == original_len:
                return f"[Error] Todo with id '{id}' not found."
            
            async with aiofiles.open(todo_file, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(todos, indent=2, ensure_ascii=False))
            return f"[Success] Deleted todo [{id}]"
            
        elif action == "toggle":
            if not id:
                return "[Error] 'id' is required for toggle action"
            
            for todo in todos:
                if todo["id"] == id:
                    if todo.get("status") == "done":
                        todo["status"] = "pending"
                        todo["completed_at"] = None
                        msg = "marked as pending"
                    else:
                        todo["status"] = "done"
                        todo["completed_at"] = datetime.now().isoformat()
                        msg = "completed"
                    
                    todo["updated_at"] = datetime.now().isoformat()
                    
                    async with aiofiles.open(todo_file, 'w', encoding='utf-8') as f:
                        await f.write(json.dumps(todos, indent=2, ensure_ascii=False))
                    
                    return f"[Success] Todo [{id}] {msg} ✅"
            
            return f"[Error] Todo with id '{id}' not found."
            
        else:
            return f"[Error] Unknown action: {action}. Use: create, list, update, delete, toggle"
            
    except Exception as e:
        return f"[Error] Todo operation failed: {str(e)}"

# ==================== 本地环境工具注册表（重命名版）====================

LOCAL_TOOLS_REGISTRY = {
    "list_files_local": {
        "type": "function",
        "function": {
            "name": "list_files_tool_local",
            "description": "List files and directories in the workspace (local filesystem).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "The directory path (default: .)"},
                    "show_all": {"type": "boolean", "description": "Show hidden files (default: false)"}
                }
            }
        }
    },
    "read_file_local": {
        "type": "function",
        "function": {
            "name": "read_file_tool_local",
            "description": "Read the contents of a file with line numbers (local filesystem).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "The path to the file"}
                },
                "required": ["path"]
            }
        }
    },
    "search_files_local": {
        "type": "function",
        "function": {
            "name": "search_files_tool_local",
            "description": "Search for a text pattern recursively in files using grep (local filesystem).",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "The regex or text to search for"},
                    "path": {"type": "string", "description": "Directory to search in (default: .)"}
                },
                "required": ["pattern"]
            }
        }
    },
    "glob_files_local": {
        "type": "function",
        "function": {
            "name": "glob_files_tool_local",
            "description": "Find files using glob patterns (local filesystem). Much more powerful than list_files for finding specific file types across the project.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string", 
                        "description": "Glob pattern like '**/*.py', 'src/**/*.ts', '*.md'"
                    },
                    "exclude": {
                        "type": "string",
                        "description": "Comma-separated exclusion patterns (default: '**/node_modules/**,**/.git/**')"
                    }
                },
                "required": ["pattern"]
            }
        }
    },
    "edit_file_local": {
        "type": "function",
        "function": {
            "name": "edit_file_tool_local",
            "description": "Create or Overwrite a file with new content (local filesystem).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "The file path"},
                    "content": {"type": "string", "description": "The full content to write to the file"}
                },
                "required": ["path", "content"]
            }
        }
    },
    "edit_file_patch_local": {
        "type": "function",
        "function": {
            "name": "edit_file_patch_tool_local",
            "description": "Precise string replacement (local filesystem). Finds a specific code block and replaces it, preserving the rest of the file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file to edit"},
                    "old_string": {"type": "string", "description": "The exact code block to replace"},
                    "new_string": {"type": "string", "description": "The new code block to insert"}
                },
                "required": ["path", "old_string", "new_string"]
            }
        }
    },
    "todo_write_local": {
        "type": "function",
        "function": {
            "name": "todo_write_tool_local",
            "description": "Task management system with persistent storage in .party/ai_todos.json (local filesystem).",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["create", "list", "update", "delete", "toggle"],
                        "description": "Operation to perform"
                    },
                    "id": {"type": "string", "description": "Task ID"},
                    "content": {"type": "string", "description": "Task description"},
                    "priority": {"type": "string", "enum": ["high", "medium", "low"]},
                    "status": {"type": "string", "enum": ["pending", "in_progress", "done", "cancelled"]}
                },
                "required": ["action"]
            }
        }
    },
    "bash_local": {
        "type": "function",
        "function": {
            "name": "bash_tool_local", 
            "description": "Execute a bash command in the local terminal. Requires 'yolo' permission mode for dangerous operations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The bash command"}
                },
                "required": ["command"]
            }
        }
    }
}

def get_local_tools_for_mode(mode: str) -> list:
    """
    根据权限模式返回本地环境工具定义列表
    """
    read_only = [
        LOCAL_TOOLS_REGISTRY["list_files_local"],
        LOCAL_TOOLS_REGISTRY["read_file_local"],
        LOCAL_TOOLS_REGISTRY["search_files_local"],
        LOCAL_TOOLS_REGISTRY["glob_files_local"]
    ]
    
    edit = [
        LOCAL_TOOLS_REGISTRY["edit_file_local"],
        LOCAL_TOOLS_REGISTRY["edit_file_patch_local"]
    ]
    
    todo = [LOCAL_TOOLS_REGISTRY["todo_write_local"]]
    bash = [LOCAL_TOOLS_REGISTRY["bash_local"]]
    
    if mode == "default":
        return read_only
    elif mode == "auto-approve": 
        return read_only + edit + todo
    elif mode == "yolo":
        return read_only + edit + todo + bash
    else:
        return read_only




# ==================== Claude Code & Qwen Code 工具（原有）=====================

cli_info = """这是一个交互式命令行工具，专门帮助用户完成软件工程任务..."""

async def claude_code_async(prompt) -> str | AsyncIterator[str]:
    """Claude Code 调用"""
    settings = await load_settings()
    CLISettings = settings.get("CLISettings", {})
    cwd = CLISettings.get("cc_path")
    ccSettings = settings.get("ccSettings", {})
    
    if not cwd or not cwd.strip():
        return "No working directory is set, please set the working directory first!"
    
    extra_config = {}
    if ccSettings.get("enabled"):
        extra_config = {
            "ANTHROPIC_BASE_URL": ccSettings.get("base_url"),
            "ANTHROPIC_API_KEY": ccSettings.get("api_key"),
            "ANTHROPIC_MODEL": ccSettings.get("model"),
        }
        extra_config = {k: str(v) if v is not None else "" for k, v in extra_config.items()}
        print(f"Using Claude Code with the following settings: {extra_config}")
    
    print(f"Using mode: {ccSettings.get('permissionMode', 'default')}")

    async def _stream() -> AsyncIterator[str]:
        options = ClaudeAgentOptions(
            cwd=cwd,
            continue_conversation=True,
            permission_mode=ccSettings.get("permissionMode", "default"),
            env={**os.environ, **extra_config}
        )
        async for message in query(prompt=prompt, options=options):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        yield block.text

    return _stream()

claude_code_tool = {
    "type": "function",
    "function": {
        "name": "claude_code_async",
        "description": f"你可以和控制CLI的智能体Claude Code进行交互。{cli_info}",
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "你想让Claude Code执行的指令...",
                }
            },
            "required": ["prompt"],
        },
    },
}

async def qwen_code_async(prompt: str) -> str | AsyncIterator[str]:
    """Qwen Code 调用"""
    settings = await load_settings()
    CLISettings = settings.get("CLISettings", {})
    cwd = CLISettings.get("cc_path")
    qcSettings = settings.get("qcSettings", {})

    if not cwd or not cwd.strip():
        return "No working directory is set, please set the working directory first!"
    
    if not os.path.isdir(cwd):
        return f"The working directory '{cwd}' does not exist!"

    extra_config: dict[str, str] = {}
    if qcSettings.get("enabled"):
        extra_config = {
            "OPENAI_BASE_URL": str(qcSettings.get("base_url") or ""),
            "OPENAI_API_KEY": str(qcSettings.get("api_key") or ""),
            "OPENAI_MODEL": str(qcSettings.get("model") or ""),
        }
    
    approval_mode = str(qcSettings.get("permissionMode", "default"))
    executable = shutil.which("qwen") or "qwen"

    async def _stream() -> AsyncIterator[str]:
        cmd_args = [executable, "-p", prompt, "--approval-mode", approval_mode]
        
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd_args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                env={**os.environ, **extra_config},
            )
        except FileNotFoundError:
            yield f"[ERROR] System cannot find the executable: {executable}..."
            return
        except Exception as e:
            yield f"[ERROR] Failed to start subprocess: {str(e)}"
            return

        async for out in _merge_streams(
            read_stream(process.stdout),
            read_stream(process.stderr, is_error=True),
        ):
            yield out

        await process.wait()

    return _stream()

qwen_code_tool = {
    "type": "function",
    "function": {
        "name": "qwen_code_async",
        "description": f"你可以和控制CLI的智能体Qwen Code进行交互。{cli_info}",
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "你想让Qwen Code执行的指令...",
                }
            },
            "required": ["prompt"],
        },
    },
}

docker_sandbox_tool = {
    "type": "function",
    "function": {
        "name": "docker_sandbox_async",
        "description": "在隔离且持久化的 Docker 沙盒环境中执行 bash 命令...",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "要执行的完整 bash 命令...",
                }
            },
            "required": ["command"],
        },
    },
}