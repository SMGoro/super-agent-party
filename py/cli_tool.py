#!/usr/bin/env python3
import asyncio
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import AsyncIterator, Union

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

# ==================== Claude Code 工具 ====================

cli_info = """这是一个交互式命令行工具，专门帮助用户完成软件工程任务。

  可以协助您：
  - 编写、调试和重构代码
  - 搜索和分析文件内容
  - 运行构建和测试
  - 管理 Git 操作
  - 代码审查和优化
  - 以及其他编程相关的任务

  运行在您的本地环境中，可以访问文件系统并使用各种工具来帮助您完成工作。

  当你被要求写一些项目或者对工作区的项目进行操作时，请尽量使用自然语言描述你的需求，这样交互式命令行工具能更好地理解并执行你的指令。
  
  你只需要给出计划，而不是具体实现，控制CLI的智能体会根据你的计划自动生成代码并执行。
"""

async def claude_code_async(prompt) -> str | AsyncIterator[str]:
    """返回 str（报错）或 AsyncIterator[str]（正常流式输出）"""
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
                    "description": "你想让Claude Code执行的指令，最好用自然语言交流，例如：请帮我创建一个文件，文件名为test.txt，文件内容为hello world",
                }
            },
            "required": ["prompt"],
        },
    },
}

# ==================== Qwen Code 工具 ====================

async def qwen_code_async(prompt: str) -> str | AsyncIterator[str]:
    """返回 str（报错）或 AsyncIterator[str]（正常流式输出）"""
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
            yield f"[ERROR] System cannot find the executable: {executable}. Is it installed and in PATH?"
            return
        except Exception as e:
            yield f"[ERROR] Failed to start subprocess: {str(e)}"
            return

        print("你的配置:", extra_config)

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
                    "description": "你想让Qwen Code执行的指令，最好用自然语言交流，例如：请帮我创建一个文件，文件名为test.txt，文件内容为hello world",
                }
            },
            "required": ["prompt"],
        },
    },
}

# ==================== Docker Sandbox 工具（已修复） ====================

import hashlib

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

def get_safe_container_name(cwd: str) -> str:
    """
    根据路径生成合法容器名
    规则：sandbox- + 路径MD5前12位（确保唯一且合法）
    """
    # 统一路径格式（绝对路径、去除尾部斜杠）
    abs_path = str(Path(cwd).resolve())
    # 生成哈希
    path_hash = hashlib.md5(abs_path.encode()).hexdigest()[:12]
    return f"sandbox-{path_hash}"

async def get_or_create_docker_sandbox(cwd: str, image_name: str = "docker/sandbox-templates:latest") -> str:
    """
    获取或创建基于路径的持久化沙盒
    返回: 容器名（同时也是沙盒ID）
    """
    container_name = get_safe_container_name(cwd)
    
    # 1. 检查容器是否已存在（包括已停止的）
    check_proc = await asyncio.create_subprocess_exec(
        "docker", "ps", "-a", "--filter", f"name=^/{container_name}$", "--format", "{{.Names}}|{{.Status}}",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    stdout, _ = await check_proc.communicate()
    output = stdout.decode().strip()
    
    if container_name in output:
        # 容器存在，检查状态
        # 格式: sandbox-xxxxx|Up 10 minutes 或 sandbox-xxxxx|Exited (0) 2 hours ago
        status = output.split("|")[-1] if "|" in output else ""
        
        if "Up" in status:
            # 已在运行，直接返回
            print(f"[INFO] 使用已运行的沙盒: {container_name}")
            return container_name
        else:
            # 已停止，启动它
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
    
    # 2. 容器不存在，创建新的
    print(f"[INFO] 创建新沙盒: {container_name} (路径: {cwd})")
    
    create_cmd = [
        "docker", "run", "-d",
        "--name", container_name,
        "-v", f"{cwd}:/workspace",  # 挂载工作目录到 /workspace
        "-w", "/workspace",         # 设置工作目录
        "--restart", "unless-stopped",  # 除非手动停止，否则自动重启
        "--label", f"sandbox.path={cwd}",  # 添加标签记录原始路径
        "--label", "sandbox.type=persistent", # 标记为持久化沙盒
        image_name,
        "tail", "-f", "/dev/null"   # 保持容器运行（不退出）
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
        # 处理并发创建冲突（race condition）
        if "is already in use by container" in error_msg:
            await asyncio.sleep(0.5)
            return await get_or_create_docker_sandbox(cwd, image_name)
        raise Exception(f"创建沙盒失败: {error_msg}")

async def docker_sandbox_async(command: str) -> str | AsyncIterator[str]:
    """
    在持久化 Docker 沙盒中执行命令
    特性：
    - 同一路径共享同一个容器（路径级隔离）
    - 自动创建/启动管理
    - 状态持久化（文件、安装的软件都会保留）
    """
    # 加载配置（假设你有类似的配置加载函数）
    settings = await load_settings()
    CLISettings = settings.get("CLISettings", {})
    cwd = CLISettings.get("cc_path")
    dsSettings = settings.get("dsSettings", {})
    
    image_name = "docker/sandbox-templates:latest"  # 或从配置读取
    
    # 验证路径
    if not cwd or not Path(cwd).is_dir():
        return f"Error: Invalid workspace directory: {cwd}"
    
    try:
        # 获取或创建沙盒（自动处理创建/启动逻辑）
        container_name = await get_or_create_docker_sandbox(cwd, image_name)
    except Exception as e:
        return f"Docker Sandbox Initialization Error: {str(e)}"

    # 返回异步生成器用于流式执行
    async def _stream() -> AsyncIterator[str]:
        # 使用 docker exec 在运行中的容器内执行命令
        # -i: 交互式（可选，如果不需要输入可去掉）
        # 使用 sh -c 来支持复杂的管道和重定向
        exec_cmd = [
            "docker", "exec",
            "-i",  # 保持 stdin 打开（支持交互式程序）
            container_name,
            "sh", "-c",
            f"cd /workspace && {command}"  # 确保在 workspace 目录执行
        ]
        
        try:
            process = await asyncio.create_subprocess_exec(
                *exec_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                # 如果你想支持输入，需要设置 stdin=asyncio.subprocess.PIPE
            )
            
            # 合并输出流
            async for line in _merge_streams(
                read_stream(process.stdout, is_error=False),
                read_stream(process.stderr, is_error=True),
            ):
                yield line
            
            await process.wait()
            
            # 可选：返回退出码提示
            if process.returncode != 0:
                yield f"[EXIT CODE] {process.returncode}"
                
        except Exception as e:
            yield f"[ERROR] 执行失败: {str(e)}"
    
    return _stream()

# 辅助函数：列出所有管理的沙盒（方便调试）
async def list_managed_sandboxes():
    """列出所有由本工具管理的沙盒"""
    proc = await asyncio.create_subprocess_exec(
        "docker", "ps", "-a", "--filter", "label=sandbox.type=persistent", 
        "--format", "table {{.Names}}\\t{{.Status}}\\t{{.Labels}}",
        stdout=asyncio.subprocess.PIPE
    )
    stdout, _ = await proc.communicate()
    return stdout.decode()

# 辅助函数：清理特定路径的沙盒
async def remove_sandbox_by_path(cwd: str):
    """删除指定路径对应的沙盒"""
    container_name = get_safe_container_name(cwd)
    proc = await asyncio.create_subprocess_exec(
        "docker", "rm", "-f", container_name,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    await proc.communicate()
    return proc.returncode == 0

# 工具定义
docker_sandbox_tool = {
    "type": "function",
    "function": {
        "name": "docker_sandbox_async",
        "description": "在隔离且持久化的 Docker 沙盒环境中执行 bash 命令。每个工作目录有独立的沙盒，环境状态（文件、安装的软件）会持久保留。自动处理沙盒的创建、启动和复用。",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "要执行的完整 bash 命令，例如 'pip install requests' 或 'ls -la'。支持管道、重定向等复杂命令。",
                }
            },
            "required": ["command"],
        },
    },
}