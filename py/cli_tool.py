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
import socket
import glob as std_glob
import fnmatch
from pathlib import Path
from typing import AsyncIterator
from datetime import datetime
from collections import deque
import aiofiles
import aiofiles.os
import hashlib
import anyio

from py.get_setting import SKILLS_DIR

# 尝试导入SDK，如果是在独立环境运行则忽略错误
try:
    from claude_agent_sdk import query, ClaudeAgentOptions, AssistantMessage, TextBlock
    from py.get_setting import load_settings
except ImportError:
    print("[WARN] SDK modules not found. Ensure 'claude_agent_sdk' and 'py.get_setting' are available.")
    # Mock load_settings for standalone testing if needed
    async def load_settings():
        return {
            "CLISettings": {"cc_path": os.getcwd()},
            "dsSettings": {},
            "localEnvSettings": {"permissionMode": "yolo"},
            "ccSettings": {"permissionMode": "default"},
            "qcSettings": {"permissionMode": "default"}
        }

# ==================== 环境初始化 ====================

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
    
    # Windows 环境简单跳过
    if platform.system() == "Windows":
        return

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
            continue
    
    print("Warning: Could not load shell environment, using current environment")

get_shell_environment()

# ==================== 核心基础设施：流处理 ====================

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

async def _get_current_cwd() -> str:
    """获取当前配置的工作目录"""
    settings = await load_settings()
    cwd = settings.get("CLISettings", {}).get("cc_path")
    if not cwd:
        raise ValueError("No workspace directory specified in settings (CLISettings.cc_path).")
    return cwd

# ==================== [新增] 核心基础设施：进程管理 ====================

class ProcessManager:
    """全局后台进程管理器 (Docker & Local) - 增强版 (支持 Windows 进程树查杀)"""
    def __init__(self):
        # 结构: {pid: {"proc": proc, "logs": deque, "cmd": str, "type": str, "task": task, "status": str, "start_time": str}}
        self._processes = {}
        self._counter = 0

    def generate_id(self):
        self._counter += 1
        return str(self._counter)

    async def register_process(self, proc, cmd: str, p_type: str):
        """注册并开始监控一个后台进程"""
        pid = self.generate_id()
        logs = deque(maxlen=2000)
        
        task = asyncio.create_task(self._monitor_output(pid, proc, logs))
        
        self._processes[pid] = {
            "proc": proc,
            "logs": logs,
            "cmd": cmd,
            "type": p_type,
            "task": task,
            "status": "running",
            "start_time": datetime.now().isoformat()
        }
        return pid

    async def _monitor_output(self, pid: str, proc, logs: deque):
        async def read_stream_to_log(stream, prefix=""):
            if not stream: return
            async for line in stream:
                decoded = line.decode('utf-8', errors='replace').rstrip()
                timestamp = datetime.now().strftime("%H:%M:%S")
                logs.append(f"[{timestamp}] {prefix}{decoded}")

        try:
            await asyncio.gather(
                read_stream_to_log(proc.stdout, ""),
                read_stream_to_log(proc.stderr, "[ERR] ")
            )
            await proc.wait()
            if pid in self._processes:
                # 只有当状态不是被手动 terminated 时才更新为 exited
                if "terminated" not in self._processes[pid]["status"]:
                    self._processes[pid]["status"] = f"exited (code {proc.returncode})"
        except Exception as e:
            if pid in self._processes:
                logs.append(f"[SYSTEM ERROR] Process monitoring failed: {str(e)}")

    def get_logs(self, pid: str, lines: int = 50) -> str:
        if pid not in self._processes:
            return f"Error: Process ID {pid} not found."
        
        entry = self._processes[pid]
        stored_logs = list(entry["logs"])
        subset = stored_logs[-lines:] if lines > 0 else stored_logs
        
        header = f"--- Logs for Process {pid} ({entry['status']}) ---\nCommand: {entry['cmd']}\n"
        return header + "\n".join(subset)

    def list_processes(self):
        if not self._processes:
            return "No background processes running."
        
        result = ["PID | Type   | Status       | Start Time          | Command"]
        result.append("-" * 90)
        
        active_found = False
        for pid, info in list(self._processes.items()):
            cmd_display = (info['cmd'][:45] + '...') if len(info['cmd']) > 45 else info['cmd']
            start_time = info['start_time'].split('T')[-1][:8]
            result.append(f"{pid:<4}| {info['type']:<7}| {info['status']:<13}| {start_time:<20}| {cmd_display}")
            active_found = True
        
        if not active_found:
            return "No background processes running."
        return "\n".join(result)

    async def kill_process(self, pid: str):
        """
        强制结束进程。
        针对 Windows 使用 taskkill /T 结束进程树，防止子进程残留。
        """
        if pid not in self._processes:
            return f"Error: Process ID {pid} not found."
        
        info = self._processes[pid]
        proc = info["proc"]
        
        # 即使 proc.returncode 已经有值，也要尝试清理可能的孤儿进程
        os_pid = proc.pid
        
        try:
            info["status"] = "terminating..."
            
            if platform.system() == "Windows":
                # Windows: 使用 taskkill /F (强制) /T (进程树) /PID <pid>
                # 这是清理 PowerShell/CMD 启动的子进程的关键
                kill_cmd = f"taskkill /F /T /PID {os_pid}"
                subprocess.run(kill_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                # Linux/Mac: 尝试杀进程组 (如果适用) 或标准 terminate
                try:
                    proc.terminate()
                    # 给一点时间优雅退出
                    await asyncio.wait_for(proc.wait(), timeout=2.0)
                except (asyncio.TimeoutError, ProcessLookupError):
                    try:
                        proc.kill()
                    except:
                        pass
            
            info["status"] = "terminated"
            return f"Process {pid} (OS PID {os_pid}) terminated successfully."
            
        except Exception as e:
            return f"Error terminating process {pid}: {str(e)}"
        
process_manager = ProcessManager()

# ==================== [新增] 核心基础设施：Docker 网络代理 ====================

class DockerPortProxy:
    """纯 Python 实现的 Docker 端口转发器 (Container -> Host)"""
    def __init__(self, container_name: str):
        self.container_name = container_name
        self.proxies = {} # {local_port: server_obj}

    async def start_forward(self, local_port: int, container_port: int):
        """开启转发：本地 TCP Server -> docker exec 桥接 -> 容器内部端口"""
        if local_port in self.proxies:
            return f"Port {local_port} is already being forwarded."

        if not self._is_port_available(local_port):
            return f"Error: Local port {local_port} is already in use."

        try:
            server = await asyncio.start_server(
                lambda r, w: self._handle_client(r, w, container_port),
                '127.0.0.1', local_port
            )
            
            self.proxies[local_port] = server
            asyncio.create_task(server.serve_forever())
            return f"Success: Forwarding localhost:{local_port} -> Docker:{container_port}"
        except Exception as e:
            return f"Error starting proxy: {str(e)}"

    def _is_port_available(self, port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('127.0.0.1', port)) != 0

    async def _handle_client(self, client_reader, client_writer, container_port):
        """处理每个连接：启动一个 docker exec 进程作为管道"""
        try:
            # 微型 Python 转发脚本，在容器内运行
            proxy_script = (
                "import socket,sys,threading;"
                "s=socket.socket();"
                f"s.connect(('127.0.0.1',{container_port}));"
                "def r():"
                " while True:"
                "  d=s.recv(4096);"
                "  if not d: break;"
                "  sys.stdout.buffer.write(d);sys.stdout.flush();\n"
                "threading.Thread(target=r,daemon=True).start();"
                "while True:"
                " d=sys.stdin.buffer.read(4096);"
                " if not d: break;"
                " s.sendall(d)"
            )

            cmd = [
                "docker", "exec", "-i", 
                self.container_name, 
                "python3", "-u", "-c", proxy_script
            ]

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL 
            )

            async def pipe_reader_to_writer(reader, writer):
                try:
                    while True:
                        data = await reader.read(4096)
                        if not data: break
                        writer.write(data)
                        await writer.drain()
                except Exception:
                    pass
                finally:
                    try: writer.close()
                    except: pass

            await asyncio.gather(
                pipe_reader_to_writer(client_reader, proc.stdin),  # Local -> Docker
                pipe_reader_to_writer(proc.stdout, client_writer)  # Docker -> Local
            )
            try: proc.terminate()
            except: pass

        except Exception as e:
            try: client_writer.close()
            except: pass

    async def stop_forward(self, local_port: int):
        if local_port in self.proxies:
            server = self.proxies[local_port]
            server.close()
            await server.wait_closed()
            del self.proxies[local_port]
            return f"Stopped forwarding on port {local_port}"
        return f"Port {local_port} was not being forwarded."
    
    def list_proxies(self):
        if not self.proxies:
            return "No active port forwardings."
        return "\n".join([f"localhost:{p} -> container:{p} (active)" for p in self.proxies.keys()])

DOCKER_PROXIES = {} # {container_name: ProxyInstance}

# ==================== Docker Sandbox 基础设施 ====================

def get_safe_container_name(cwd: str) -> str:
    """根据路径生成合法容器名"""
    abs_path = str(Path(cwd).resolve())
    path_hash = hashlib.md5(abs_path.encode()).hexdigest()[:12]
    return f"sandbox-{path_hash}"

async def get_or_create_docker_sandbox(cwd: str, image_name: str = "docker/sandbox-templates:claude-code") -> str:
    """获取或创建基于路径的持久化沙盒，并映射全局skills目录"""
    container_name = get_safe_container_name(cwd)
    
    # 获取主机的全局skills目录
    host_skills_dir = SKILLS_DIR
    
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
            return container_name
        else:
            # 启动已存在的容器
            await asyncio.create_subprocess_exec("docker", "start", container_name, stdout=asyncio.subprocess.PIPE)
            return container_name
    
    # 创建新容器，映射主机的全局skills目录
    # 注意：我们将主机skills目录映射到容器内的 /root/.agents/skills
    # 这是标准Agent Skills CLI使用的路径
    create_cmd = [
        "docker", "run", "-d",
        "--name", container_name,
        "-v", f"{cwd}:/workspace",  # 映射工作目录
        "-v", f"{host_skills_dir}:/home/agent/.agents/skills",   # 映射全局skills目录到容器内
        "-w", "/workspace",
        "--restart", "unless-stopped",
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
        # 容器创建成功，确保容器内的skills目录权限正确
        try:
            # 设置容器内skills目录的权限
            chown_cmd = [
                "docker", "exec", container_name,
                "chown", "-R", "root:root", "/root/.agents/skills"
            ]
            chown_proc = await asyncio.create_subprocess_exec(
                *chown_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await chown_proc.communicate()
        except Exception:
            # 权限设置失败不影响主要功能
            pass
        
        return container_name
    else:
        # 简单重试逻辑
        if "is already in use" in stderr.decode():
            await asyncio.sleep(0.5)
            return await get_or_create_docker_sandbox(cwd, image_name)
        raise Exception(f"Failed to create sandbox: {stderr.decode()}")


async def _exec_docker_cmd_simple(cwd: str, cmd_list: list) -> str:
    """内部辅助函数：在容器内执行简单命令并获取输出"""
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

# ==================== Docker 环境工具实现 (含新功能) ====================

async def docker_sandbox_async(command: str, background: bool = False) -> str | AsyncIterator[str]:
    """
    [Docker] 在沙盒中执行命令
    新增参数: background (True则后台运行并返回PID)
    """
    settings = await load_settings()
    cwd = settings.get("CLISettings", {}).get("cc_path")
    if not cwd: return "Error: No workspace directory specified in settings."
    
    try:
        container_name = await get_or_create_docker_sandbox(cwd)
    except Exception as e:
        return f"Docker Sandbox Error: {str(e)}"

    exec_cmd = [
        "docker", "exec",
        "-i", # 保持stdin打开对某些交互式命令很重要
        container_name,
        "sh", "-c",
        f"cd /workspace && {command}"
    ]
    
    try:
        process = await asyncio.create_subprocess_exec(
            *exec_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        # === 后台模式 ===
        if background:
            pid = await process_manager.register_process(process, f"[Docker] {command}", "docker")
            return f"[SUCCESS] Docker background process started.\nPID: {pid}\nContainer: {container_name}\nUse 'manage_processes' to view logs."

        # === 前台模式 (流式) ===
        async def _stream() -> AsyncIterator[str]:
            output_yielded = False
            async for line in _merge_streams(
                read_stream(process.stdout, is_error=False),
                read_stream(process.stderr, is_error=True),
            ):
                yield line
                output_yielded = True
            
            await process.wait()
            if process.returncode != 0:
                yield f"[EXIT CODE] {process.returncode}"
            elif not output_yielded:
                yield "[SUCCESS] 命令已成功执行未报错"
    
        return _stream()
    except Exception as e:
        return f"[ERROR] Execution failed: {str(e)}"

async def edit_file_patch_tool(path: str, old_string: str, new_string: str) -> str:
    """[Docker] 精确字符串替换"""
    try:
        real_cwd = await _get_current_cwd()
        container_name = await get_or_create_docker_sandbox(real_cwd)
        
        content = await _exec_docker_cmd_simple(real_cwd, ["cat", path])
        
        normalized_content = "\n".join(line.rstrip() for line in content.split("\n"))
        normalized_old = "\n".join(line.rstrip() for line in old_string.split("\n"))
        
        if normalized_old not in normalized_content:
            lines = content.split("\n")
            first_line = old_string.split("\n")[0] if "\n" in old_string else old_string
            similar_lines = [f"Line {i+1}: {line[:80]}" for i, line in enumerate(lines) if first_line.strip() in line]
            error_msg = f"[Error] Old string not found in file '{path}'.\n"
            if similar_lines:
                error_msg += f"\nFound similar lines:\n" + "\n".join(similar_lines[:5])
            return error_msg
        
        new_content = content.replace(old_string, new_string, 1)
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8') as tmp:
            tmp.write(new_content)
            tmp_path = tmp.name
        
        dest_path = f"{container_name}:/workspace/{path}"
        cp_proc = await asyncio.create_subprocess_exec("docker", "cp", tmp_path, dest_path, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        await cp_proc.communicate()
        os.unlink(tmp_path)
        
        if cp_proc.returncode != 0: return "[Error] Patch copy failed."
        return f"[Success] Patched '{path}'."
        
    except Exception as e:
        return f"[Error] Patch failed: {str(e)}"

async def glob_files_tool(pattern: str, exclude: str = "**/node_modules/**,**/.git/**,**/__pycache__/**") -> str:
    """[Docker] Glob 递归查找"""
    try:
        real_cwd = await _get_current_cwd()
        exclude_list = [e.strip() for e in exclude.split(",") if e.strip()]
        
        python_script = f'''
import glob, os, json, fnmatch
files = glob.glob("/workspace/{pattern}", recursive=True)
exclude_patterns = {exclude_list}
filtered = []
for f in files:
    if not os.path.isfile(f): continue
    rel_path = f.replace("/workspace/", "")
    should_exclude = False
    for ex in exclude_patterns:
        if fnmatch.fnmatch(rel_path, ex) or fnmatch.fnmatch(f, ex):
            should_exclude = True; break
    if not should_exclude: filtered.append(rel_path)
print(json.dumps(filtered))
'''
        output = await _exec_docker_cmd_simple(real_cwd, ["python3", "-c", python_script])
        files = json.loads(output)
        if not files: return "[Result] No files found."
        
        lines = [f"[{len(files)} files matched]"]
        for f in files[:50]:
            icon = "🐍" if f.endswith(".py") else "📄"
            lines.append(f"{icon} {f}")
        if len(files) > 50: lines.append(f"... {len(files)-50} more")
        return "\n".join(lines)
    except Exception as e:
        return f"[Error] Glob failed: {str(e)}"

async def todo_write_tool(action: str, id: str = None, content: str = None, priority: str = "medium", status: str = None) -> str:
    """[Docker] 任务管理"""
    try:
        real_cwd = await _get_current_cwd()
        container_name = await get_or_create_docker_sandbox(real_cwd)
        todo_file = "/workspace/.agent/ai_todos.json"
        
        try:
            data = await _exec_docker_cmd_simple(real_cwd, ["cat", todo_file])
            todos = json.loads(data)
        except:
            todos = []
            
        if action == "create":
            if not content: return "[Error] Content required."
            new_todo = {
                "id": id or str(uuid.uuid4())[:8],
                "content": content,
                "priority": priority,
                "status": "pending",
                "created_at": datetime.now().isoformat(),
                "completed_at": None  # 初始化完成时间
            }
            todos.append(new_todo)
            msg = f"[Success] Created {new_todo['id']}"
            
        elif action == "list":
            if not todos: return "No todos."
            lines = ["📋 Tasks:"]
            # 排序逻辑：未完成在前，高优先级在前
            for t in sorted(todos, key=lambda x: (x.get('status') == 'done', x.get('priority') != 'high')):
                icon = "✅" if t.get('status') == 'done' else "⏳"
                lines.append(f"{icon} [{t['id']}] {t['content'][:40]}")
            return "\n".join(lines)
            
        elif action in ["update", "toggle", "delete"]:
            if not id: return "[Error] ID required."
            target = next((t for t in todos if t['id'] == id), None)
            if not target: return f"ID {id} not found."
            
            if action == "delete":
                todos.remove(target)
                msg = f"Deleted {id}"

            elif action == "toggle":
                # 核心逻辑：切换状态并记录/重置完成时间
                if target.get('status') != 'done':
                    target['status'] = 'done'
                    target['completed_at'] = datetime.now().isoformat()
                else:
                    target['status'] = 'pending'
                    target['completed_at'] = None
                msg = f"Toggled {id} to {target['status']}"

            elif action == "update":
                if content: target['content'] = content
                if priority: target['priority'] = priority
                
                # 核心逻辑：如果 status 被明确修改
                if status:
                    if status == "done" and target.get('status') != "done":
                        target['completed_at'] = datetime.now().isoformat()
                    elif status != "done" and target.get('status') == "done":
                        target['completed_at'] = None
                    target['status'] = status
                
                target['updated_at'] = datetime.now().isoformat()
                msg = f"Updated {id}"
        else:
            return "Unknown action."

        # 写回逻辑 (保持不变)
        with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8') as tmp:
            tmp.write(json.dumps(todos, indent=2, ensure_ascii=False))
            tmp_path = tmp.name
        
        await _exec_docker_cmd_simple(real_cwd, ["mkdir", "-p", "/workspace/.agent"])
        dest = f"{container_name}:{todo_file}"
        proc = await asyncio.create_subprocess_exec("docker", "cp", tmp_path, dest, stdout=asyncio.subprocess.PIPE)
        await proc.wait()
        
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
            
        return msg
    except Exception as e:
        return f"[Error] Todo failed: {str(e)}"
# 恢复原有的 Docker 基础文件工具
async def list_files_tool(path: str = ".", show_all: bool = True) -> str:
    try:
        real_cwd = await _get_current_cwd()
        flag = "-laF" if show_all else "-F"
        return await _exec_docker_cmd_simple(real_cwd, ["ls", flag, path])
    except Exception as e: return str(e)

async def read_file_tool(path: str) -> str:
    """[Docker] 读取文件：增加大小限制和结构化提示"""
    try:
        real_cwd = await _get_current_cwd()
        # 使用 shell 脚本限制读取前 2000 行，并返回总行数提示
        script = f"""
        if [ ! -f "{path}" ]; then echo "[Error] File not found: {path}"; exit 0; fi
        total=$(wc -l < "{path}" 2>/dev/null || echo 0)
        head -n 2000 "{path}" | cat -n
        if [ "$total" -gt 2000 ]; then
            echo ""
            echo "... [Warning] File truncated (Too large). Showing 1 to 2000 of $total lines."
            echo "💡 [Next Step Hint] Use 'read_file_range' to read specific lines (e.g. start: 2001, end: 2500) or 'tail_file' to read the end of the log."
        fi
        """
        return await _exec_docker_cmd_simple(real_cwd, ["sh", "-c", script])
    except Exception as e: return str(e)

async def read_file_range_tool(path: str, start_line: int, end_line: int) -> str:
    """[Docker] 精准读取文件指定行范围"""
    try:
        if start_line < 1 or end_line < start_line:
            return "[Error] Invalid line range."
        real_cwd = await _get_current_cwd()
        # 使用 awk 高效读取指定行，并带上行号
        script = f"""awk 'NR>={start_line} && NR<={end_line} {{printf "%5d | %s\\n", NR, $0}}' "{path}" """
        return await _exec_docker_cmd_simple(real_cwd, ["sh", "-c", script])
    except Exception as e: return str(e)

async def tail_file_tool(path: str, lines: int = 100) -> str:
    """[Docker] 读取文件末尾（常用于日志）"""
    try:
        real_cwd = await _get_current_cwd()
        # 先打行号，再 tail
        script = f"""cat -n "{path}" | tail -n {lines}"""
        return await _exec_docker_cmd_simple(real_cwd, ["sh", "-c", script])
    except Exception as e: return str(e)

async def edit_file_tool(path: str, content: str) -> str:
    try:
        real_cwd = await _get_current_cwd()
        container_name = await get_or_create_docker_sandbox(real_cwd)
        with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8') as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        await _exec_docker_cmd_simple(real_cwd, ["mkdir", "-p", os.path.dirname(path) or "."])
        dest = f"{container_name}:/workspace/{path}"
        proc = await asyncio.create_subprocess_exec("docker", "cp", tmp_path, dest, stdout=asyncio.subprocess.PIPE)
        await proc.wait()
        os.unlink(tmp_path)
        return f"[Success] Saved {path}"
    except Exception as e: return str(e)

async def search_files_tool(pattern: str, path: str = ".") -> str:
    try:
        real_cwd = await _get_current_cwd()
        return await _exec_docker_cmd_simple(real_cwd, ["grep", "-rn", pattern, path])
    except Exception as e: return str(e)


# ==================== [新增] 管理工具：进程与网络 ====================

async def manage_processes_tool(action: str, pid: str = None) -> str:
    """[Common] 管理后台进程"""
    if action == "list":
        return process_manager.list_processes()
    if action == "logs":
        if not pid: return "Error: 'pid' is required for logs."
        return process_manager.get_logs(pid)
    if action == "kill":
        if not pid: return "Error: 'pid' is required for kill."
        return await process_manager.kill_process(pid)
    return "Error: Unknown action. Use list, logs, or kill."

async def docker_manage_ports_tool(action: str, container_port: int = 8000, host_port: int = None) -> str:
    """[Docker] 端口转发管理"""
    try:
        real_cwd = await _get_current_cwd()
        container_name = await get_or_create_docker_sandbox(real_cwd)
        
        if container_name not in DOCKER_PROXIES:
            DOCKER_PROXIES[container_name] = DockerPortProxy(container_name)
        proxy = DOCKER_PROXIES[container_name]
        
        if action == "list":
            return proxy.list_proxies()
        if action == "forward":
            if not host_port: host_port = container_port
            return await proxy.start_forward(host_port, container_port)
        if action == "stop":
            if not host_port: return "Error: host_port required to stop."
            return await proxy.stop_forward(host_port)
        return "Unknown action."
    except Exception as e:
        return f"[Error] Port tool failed: {str(e)}"

async def local_net_tool(action: str, port: int = None) -> str:
    """[Local] 本地网络工具：检查端口占用"""
    if action == "check":
        if not port: return "Error: Port required."
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            result = s.connect_ex(('127.0.0.1', port))
            status = "OPEN/BUSY" if result == 0 else "CLOSED/FREE"
            return f"Port {port} on localhost is {status}."
    
    if action == "scan":
        # 简单扫描常用开发端口
        common_ports = [3000, 5000, 8000, 8080, 80, 443, 3306, 5432]
        results = []
        for p in common_ports:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(0.1)
                res = s.connect_ex(('127.0.0.1', p))
                status = "BUSY" if res == 0 else "FREE"
                results.append(f"{p}: {status}")
        return "Common Ports:\n" + "\n".join(results)
        
    return "Unknown action. Use check or scan."

# ==================== 本地环境 (Local) 工具实现 ====================

def resolve_strict_path(cwd: str, sub_path: str, check_symlink: bool = True) -> Path:
    """
    严格工作区路径解析
    - 禁止绝对路径
    - 禁止 ../ 遍历  
    - 禁止通过符号链接指向工作区外
    """
    base = Path(cwd).resolve()
    
    if not sub_path:
        return base
        
    # 清理输入（阻止空字节、换行等）
    sub_path = sub_path.strip().replace('\x00', '').replace('\n', '')
    
    # 显式禁止路径遍历模式（快速失败）
    if '..' in sub_path.split(os.sep):
        raise PermissionError(f"Path traversal detected: {sub_path}")
    
    # 禁止绝对路径（Windows C:\ 和 Unix /）
    if os.path.isabs(sub_path) or (len(sub_path) > 1 and sub_path[1] == ':'):
        raise PermissionError(f"Absolute paths not allowed: {sub_path}")
    
    # 解析完整路径
    target = (base / sub_path).resolve()
    
    # 关键检查：确保 resolve 后的路径仍在 base 内
    try:
        target.relative_to(base)
    except ValueError:
        raise PermissionError(f"Access denied: {sub_path} resolves outside workspace")
    
    # 符号链接检查（防止 /workspace/link -> /etc）
    if check_symlink and target.exists():
        real_path = target.resolve(strict=True)
        try:
            real_path.relative_to(base)
        except ValueError:
            raise PermissionError(f"Symlink escape detected: {sub_path} -> {real_path}")
            
    return target

from typing import Tuple

def validate_bash_command(command: str, cwd: str, mode: str = "default") -> Tuple[bool, str]:
    """
    安全校验策略（增强版）：
    1. 严格禁止路径遍历（../）
    2. 禁止访问敏感系统目录
    3. 允许正常的文件操作和相对路径
    """
    
    # ===== 1. 路径遍历防御 (核心修复) =====
    # 解析：
    # (?:\s|^|/)  -> 前面必须是 空格、开头、或斜杠 (防止匹配到 valid..file 这种文件名)
    # \.\.        -> 也就是 ".."
    # (?:/|\s|$)  -> 后面必须是 斜杠、空格、或结尾 (防止匹配到 ..valid 这种文件名)
    # 结果：能拦截 "../", "/..", " .. ", "ls .."，但不会误伤 "echo loading..."
    traversal_pattern = r'(?:\s|^|/)\.\.(?:/|\s|$)'
    
    if re.search(traversal_pattern, command):
        return False, "Path traversal detected (usage of '..' is blocked to keep agent in workspace)"

    # ===== 2. 绝对路径与敏感目录防御 =====
    # 常见敏感目录前缀
    sensitive_roots = [
        r'/etc', r'/var', r'/root', r'/bin', r'/sbin', r'/usr',  # Linux
        r'C:\\Windows', r'C:\\Program Files', r'C:\\Users'       # Windows
    ]
    
    # 检查是否直接操作了敏感路径
    for root in sensitive_roots:
        # 匹配 空格+路径 或 开头+路径
        # 例如: "cat /etc/passwd" 或 "/etc/init.d/..."
        if re.search(r'(?:\s|^)' + root, command, re.IGNORECASE):
            return False, f"Access to system directory '{root}' is blocked"

    # 针对 cd 命令的额外保护：禁止 cd /... (除非是 /workspace 或 /tmp)
    # 拦截: cd /etc, cd /
    # 放行: cd ., cd subfolder, cd /workspace
    if re.search(r'\bcd\s+/(?!(workspace|tmp|dev/null))', command, re.IGNORECASE):
            return False, "Changing directory to outside workspace is blocked"

    # ===== 3. 毁灭性操作（保持原样）=====
    destructive_patterns = [
        (r'rm\s+-rf\s*/', "Recursive delete root"),                
        (r'mkfs\.[a-z]+', "Filesystem format"),                    
        (r'dd\s+if=.*of=/dev/[a-z]', "Direct device write"),       
        (r'>?\s*/dev/(sda|hd|nvme|mmcblk)', "Block device access"),
        (r':\(\)\{\s*:\|:&?\s*\};\s*:', "Fork bomb"), # 新增 Fork bomb 防御
    ]
    
    for pattern, reason in destructive_patterns:
        if re.search(pattern, command, re.IGNORECASE):
            return False, f"Destructive operation blocked: {reason}"
    
    # ===== 4. 风险操作（非 YOLO 模式拦截）=====
    if mode != "yolo":
        risk_patterns = [
            # 允许 wget/curl，但禁止直接管道给 shell 执行 (如 curl | sh)
            (r'(curl|wget).*\|\s*(sh|bash|zsh|python|perl|php)', "Remote execution via pipe"),
            # 禁止访问环境变量 HOME，防止泄露主机用户目录
            (r'\$\{?HOME\}?', "HOME env variable usage"),
            (r'~\s*/', "Home directory access via ~"),
        ]
        for pattern, reason in risk_patterns:
            if re.search(pattern, command, re.IGNORECASE):
                return False, f"{reason} blocked in {mode} mode"
    
    return True, command

# ===== 修复乱码：增加 GBK 解码支持 =====
async def read_stream(stream, *, is_error: bool = False):
    """读取流并添加错误前缀，支持 Windows 中文编码"""
    if stream is None:
        return
    async for line in stream:
        prefix = "[ERROR] " if is_error else ""
        
        # Windows 中文系统通常用 GBK，先尝试 UTF-8，失败则尝试 GBK
        try:
            decoded = line.decode('utf-8').rstrip()
        except UnicodeDecodeError:
            try:
                decoded = line.decode('gbk').rstrip()
            except:
                decoded = line.decode('utf-8', errors='replace').rstrip()
                
        yield f"{prefix}{decoded}"


async def bash_tool_local(command: str, background: bool = False) -> str | AsyncIterator[str]:
    """[Local] 执行命令，支持后台"""
    settings = await load_settings()
    cwd = settings.get("CLISettings", {}).get("cc_path")
    perm = settings.get("localEnvSettings", {}).get("permissionMode", "default")
    
    if not cwd: 
        return "Error: No workspace."
    
    # 安全检查（不再包装 cd 命令）
    allowed, result = validate_bash_command(command, cwd, mode=perm)
    if not allowed:
        return f"[Security] Command blocked: {result}"
    
    # 保持和原版完全一致：不修改 command，只检查

    system = platform.system()
    if system == "Windows":
        is_ps = any(x in command.lower() for x in ['get-', 'set-location', 'select-string'])
        exe = "powershell.exe" if is_ps else "cmd.exe"
        args = ["-Command", command] if is_ps else ["/c", command]
    else:
        exe = os.environ.get('SHELL', '/bin/bash')
        args = ["-c", command]

    try:
        proc = await asyncio.create_subprocess_exec(
            exe, *args,
            stdout=asyncio.subprocess.PIPE, 
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,  # ← 原版逻辑：靠这个设置目录，不在命令里 cd
            env=os.environ.copy()
        )

        if background:
            pid = await process_manager.register_process(proc, command, "local")
            return f"[SUCCESS] Background process started.\nPID: {pid}\nUse 'manage_processes_local' to check."

        async def _stream():
            yielded = False
            async for line in _merge_streams(read_stream(proc.stdout), read_stream(proc.stderr, is_error=True)):
                yield line
                yielded = True
            await proc.wait()
            if proc.returncode != 0: 
                yield f"[EXIT] {proc.returncode}"
            elif not yielded: 
                yield "[SUCCESS] No output."
        return _stream()
    except Exception as e: 
        return str(e)

# 恢复原有的 Local 文件工具
async def list_files_tool_local(path: str = ".", show_all: bool = True) -> str:
    """[Local] 列出文件：优先显示目录，支持数量截断，过滤隐藏文件"""
    try:
        cwd = await _get_current_cwd()
        target = resolve_strict_path(cwd, path, check_symlink=True)
        
        if not target.is_dir():
            return f"[Error] Not a directory: {path}"

        # 使用 scandir 获取更详细的信息且速度更快
        entries = []
        try:
            with os.scandir(target) as it:
                for entry in it:
                    if not show_all and entry.name.startswith('.'):
                        continue
                    
                    is_dir = entry.is_dir()
                    # 格式：(是否目录, 排序名, 显示字符串)
                    # 目录排在前面 (0)，文件排在后面 (1)
                    display_name = f"{entry.name}/" if is_dir else entry.name
                    entries.append((0 if is_dir else 1, entry.name.lower(), display_name))
        except PermissionError:
            return f"[Error] Permission denied accessing: {path}"

        # 排序：先按目录/文件分，再按名称字母序
        entries.sort()

        # 数量截断防止 Token 爆炸
        MAX_ITEMS = 200
        result_lines = [e[2] for e in entries[:MAX_ITEMS]]
        
        summary = f"Total: {len(entries)} items"
        if len(entries) > MAX_ITEMS:
            summary += f" (Showing first {MAX_ITEMS})"
            result_lines.append(f"... {len(entries) - MAX_ITEMS} more items")
        
        return f"{summary} in {path}:\n" + "\n".join(result_lines) if result_lines else "Empty directory."

    except Exception as e:
        return f"[Error] List failed: {str(e)}"

async def read_file_tool_local(path: str) -> str:
    """[Local] 读取文件：支持大文件截断读取，并返回结构化下一步建议"""
    try:
        cwd = await _get_current_cwd()
        target = resolve_strict_path(cwd, path, check_symlink=True)

        if not target.exists() or not target.is_file():
            return f"[Error] File not found or not a file: {path}"

        try:
            with open(target, 'rb') as f_bin:
                if b'\0' in f_bin.read(1024):
                    return f"[Error] Cannot read binary file: {path}"
        except Exception as e:
            return f"[Error] Failed to check file type: {str(e)}"

        MAX_LINES = 2000
        MAX_BYTES = 500 * 1024  
        file_size = target.stat().st_size
        truncated = False
        
        async with aiofiles.open(target, 'r', encoding='utf-8', errors='replace') as f:
            if file_size > MAX_BYTES:
                content = await f.read(MAX_BYTES)
                truncated = True
                lines = content.splitlines()
                if lines: lines.pop()
            else:
                lines = await f.readlines()
                lines = [l.rstrip('\n') for l in lines]

        if len(lines) > MAX_LINES:
            lines = lines[:MAX_LINES]
            truncated = True

        output = [f"{i+1:4} | {line}" for i, line in enumerate(lines)]
        
        if truncated:
            output.append(f"\n... [Warning] File content truncated (Too large). Showing first {len(lines)} lines.")
            output.append(f"💡 [Next Step Hint] The file is large. Use 'read_file_range_local' to read lines {len(lines)+1} to {len(lines)+500}, or 'tail_file_local' to view the end.")
            
        return "\n".join(output)
    except Exception as e: return f"[Error] Read failed: {str(e)}"

async def read_file_range_tool_local(path: str, start_line: int, end_line: int) -> str:
    """[Local] 精准读取文件指定行范围"""
    try:
        if start_line < 1 or end_line < start_line:
            return "[Error] Invalid line range. start_line must be >= 1 and end_line >= start_line."
            
        cwd = await _get_current_cwd()
        target = resolve_strict_path(cwd, path, check_symlink=True)
        
        if not target.exists() or not target.is_file(): return f"[Error] File not found: {path}"

        async with aiofiles.open(target, 'r', encoding='utf-8', errors='replace') as f:
            lines = await f.readlines()
            
        if start_line > len(lines):
            return f"[Error] start_line ({start_line}) is beyond file length ({len(lines)})."
            
        subset = lines[start_line - 1 : end_line]
        return "\n".join(f"{i + start_line:4} | {line.rstrip('\n')}" for i, line in enumerate(subset))
    except Exception as e: return f"[Error] Range read failed: {str(e)}"

async def tail_file_tool_local(path: str, lines: int = 100) -> str:
    """[Local] 读取文件末尾（常用于日志）"""
    try:
        cwd = await _get_current_cwd()
        target = resolve_strict_path(cwd, path, check_symlink=True)
        if not target.exists() or not target.is_file(): return f"[Error] File not found: {path}"

        # 本地简单实现：读入后切片（如果文件极大建议改用 seek 倒序读，但此处通常够用）
        async with aiofiles.open(target, 'r', encoding='utf-8', errors='replace') as f:
            all_lines = await f.readlines()
            
        subset = all_lines[-lines:] if lines < len(all_lines) else all_lines
        start_idx = max(1, len(all_lines) - lines + 1)
        
        return "\n".join(f"{i + start_idx:4} | {line.rstrip('\n')}" for i, line in enumerate(subset))
    except Exception as e: return f"[Error] Tail failed: {str(e)}"

async def edit_file_tool_local(path: str, content: str) -> str:
    """[Local] 写入文件：修复了绝对路径误判问题"""
    try:
        cwd = await _get_current_cwd()
        # 这一步已经确保了 path 不会逃逸出 cwd
        target = resolve_strict_path(cwd, path, check_symlink=True)
        
        # 1. 确保父目录存在
        parent_dir = target.parent
        # --- 删除了导致报错的 resolve_strict_path(cwd, str(parent_dir)...) ---
        
        await aiofiles.os.makedirs(parent_dir, exist_ok=True)

        # 2. 创建备份 (如果文件存在)
        backup_msg = ""
        if target.exists():
            try:
                backup_path = target.with_suffix(target.suffix + ".bak")
                shutil.copy2(target, backup_path)
                backup_msg = f" (Backup created: {backup_path.name})"
            except Exception as e:
                print(f"[Warn] Backup failed: {e}")

        # 3. 原子写入
        temp_path = target.with_suffix(target.suffix + f".tmp.{uuid.uuid4().hex[:6]}")
        try:
            async with aiofiles.open(temp_path, 'w', encoding='utf-8') as f:
                await f.write(content)
            
            if os.path.exists(target):
                os.replace(temp_path, target)
            else:
                os.rename(temp_path, target)
        except Exception as e:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise e

        return f"Saved successfully{backup_msg}."

    except Exception as e:
        return f"[Error] Edit failed: {str(e)}"

async def search_files_tool_local(pattern: str, path: str = ".") -> str:
    """[Local] 智能搜索：优先尝试 git grep/grep，回退到优化的 Python 实现"""
    try:
        cwd = await _get_current_cwd()
        target_dir = resolve_strict_path(cwd, path, check_symlink=True)
        target_str = str(target_dir)
        
        # 1. 尝试使用 git grep (速度最快，且自动尊重 .gitignore)
        # 只有当在 git 仓库内且安装了 git 时有效
        if os.path.isdir(os.path.join(cwd, ".git")) and shutil.which("git"):
            try:
                # -I: 不搜索二进制, -n: 行号, --full-name: 相对路径
                cmd = ["git", "grep", "-I", "-n", "--full-name", pattern]
                # 如果指定了子目录，限制搜索范围
                rel_path = os.path.relpath(target_str, cwd)
                if rel_path != ".":
                    cmd.append(rel_path)
                
                proc = await asyncio.create_subprocess_exec(
                    *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, cwd=cwd
                )
                stdout, _ = await proc.communicate()
                if proc.returncode == 0 and stdout:
                    return stdout.decode('utf-8', errors='replace').strip()
            except Exception:
                pass # git grep 失败则回退

        # 2. 优化的 Python 实现 (Ripgrep-lite)
        matches = []
        regex = re.compile(pattern)
        MAX_RESULTS = 1000  # 防止结果爆炸
        
        # 定义需要跳过的目录和扩展名
        SKIP_DIRS = {'.git', 'node_modules', '__pycache__', 'venv', '.env', 'dist', 'build', 'coverage'}
        SKIP_EXTS = {'.pyc', '.pyo', '.so', '.dll', '.exe', '.bin', '.png', '.jpg', '.jpeg', '.gif', '.zip', '.tar', '.gz'}

        # 判断文件是否为二进制 (读取前 1024 字节检查 NULL)
        def is_binary(file_path):
            try:
                with open(file_path, 'rb') as f:
                    chunk = f.read(1024)
                    return b'\0' in chunk
            except:
                return True

        for root, dirs, files in os.walk(target_str, topdown=True):
            # 剪枝：直接修改 dirs 列表，阻止 os.walk 进入这些目录
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith('.')]
            
            for file in files:
                if any(file.endswith(ext) for ext in SKIP_EXTS): continue
                
                full_path = os.path.join(root, file)
                # 相对路径用于显示
                display_path = os.path.relpath(full_path, cwd)
                
                if is_binary(full_path): continue

                try:
                    # 使用 aiofiles 异步读取文本
                    async with aiofiles.open(full_path, 'r', encoding='utf-8', errors='replace') as f:
                        content = await f.read()
                        lines = content.splitlines()
                        for i, line in enumerate(lines, 1):
                            if regex.search(line):
                                # 截断过长的行
                                clean_line = line.strip()[:200]
                                matches.append(f"{display_path}:{i}:{clean_line}")
                                if len(matches) >= MAX_RESULTS:
                                    return "\n".join(matches) + f"\n... (Truncated at {MAX_RESULTS} matches)"
                except Exception:
                    continue

        return "\n".join(matches) if matches else "No matches found."
    except Exception as e:
        return f"[Error] Search failed: {str(e)}"
    
async def glob_files_tool_local(pattern: str, exclude: str = "") -> str:
    """[Local] 智能查找：修复了拦截 '..' 的过度限制"""
    try:
        cwd = await _get_current_cwd()
        base = Path(cwd).resolve()
        
        # 移除原有的 if '..' in pattern 拦截逻辑
        # 依靠后续的 Path(root).relative_to(base) 来确保安全

        excludes = [e.strip() for e in exclude.split(",") if e.strip()]
        DEFAULT_EXCLUDES = {'.git', 'node_modules', '__pycache__', 'venv', 'dist', 'build'}
        
        results = []

        # 1. 尝试使用 git ls-files (略过，逻辑同原版)
        # ... (中间 git 逻辑保持不变) ...

        # 2. 优化的遍历逻辑
        for root, dirs, files in os.walk(str(base), topdown=True):
            # 剪枝
            dirs[:] = [d for d in dirs if d not in DEFAULT_EXCLUDES and not d.startswith('.')]
            
            try:
                # 核心安全检查：确保当前遍历到的 root 仍在 base 内部
                rel_root = Path(root).relative_to(base)
            except ValueError:
                continue # 如果越界了，跳过该目录

            for name in files:
                file_rel_path = str(rel_root / name)
                if file_rel_path.startswith("./"): file_rel_path = file_rel_path[2:]

                if any(fnmatch.fnmatch(file_rel_path, ex) for ex in excludes):
                    continue
                
                # 检查匹配项
                if fnmatch.fnmatch(file_rel_path, pattern):
                    results.append(file_rel_path)

        limit = 200
        output = sorted(results)
        if len(output) > limit:
            return "\n".join(output[:limit]) + f"\n... ({len(output)-limit} more files)"
        return "\n".join(output) if output else "No files matched."
        
    except Exception as e:
        return f"[Error] Glob failed: {str(e)}"

async def edit_file_patch_tool_local(path: str, old_string: str, new_string: str) -> str:
    """[Local] 精确替换：自动处理换行符差异 (CRLF/LF) 与空白字符容错"""
    try:
        cwd = await _get_current_cwd()
        target = resolve_strict_path(cwd, path, check_symlink=True)
        
        if not target.exists():
            return f"[Error] File not found: {path}"

        # 读取文件内容
        async with aiofiles.open(target, 'r', encoding='utf-8') as f:
            content = await f.read()

        # --- 策略 1: 直接替换 (最快) ---
        if old_string in content:
            new_content = content.replace(old_string, new_string, 1)
            async with aiofiles.open(target, 'w', encoding='utf-8') as f:
                await f.write(new_content)
            return "Patched successfully (Exact match)."

        # --- 策略 2: 归一化换行符后替换 (处理 Windows/Linux 差异) ---
        # 将所有 \r\n 转换为 \n 进行比对
        content_normalized = content.replace('\r\n', '\n')
        old_normalized = old_string.replace('\r\n', '\n')
        new_normalized = new_string.replace('\r\n', '\n')

        if old_normalized in content_normalized:
            # 这里的难点是：如果我们在 normalized 版本中替换了，
            # 我们需要把写回的内容最好保持原文件的换行符风格。
            # 简单起见，我们统一写回 normalized 的内容 (Python write 通常会自动处理 OS 换行)
            new_content_normalized = content_normalized.replace(old_normalized, new_normalized, 1)
            async with aiofiles.open(target, 'w', encoding='utf-8') as f:
                await f.write(new_content_normalized)
            return "Patched successfully (Normalized line endings match)."

        # --- 策略 3: 容错匹配 (忽略行尾空格) ---
        # 如果还是找不到，尝试逐行对比，忽略 strip() 后的差异
        lines = content.splitlines()
        old_lines = old_string.splitlines()
        
        if not old_lines: return "[Error] old_string is empty."

        # 简单的滑动窗口匹配
        match_index = -1
        for i in range(len(lines) - len(old_lines) + 1):
            match = True
            for j in range(len(old_lines)):
                if lines[i+j].strip() != old_lines[j].strip():
                    match = False
                    break
            if match:
                match_index = i
                break
        
        if match_index != -1:
            # 找到了逻辑上匹配的块，进行替换
            # 注意：这里我们使用 new_string (保持 AI 生成的格式)
            # 但我们需要小心缩进。这里假设 AI 提供了正确的 new_string 缩进。
            pre_content = "\n".join(lines[:match_index])
            post_content = "\n".join(lines[match_index + len(old_lines):])
            
            # 拼接时要注意原文件的换行符，这里简化为 \n
            final_content = (pre_content + "\n" + new_string + "\n" + post_content).strip()
            
            async with aiofiles.open(target, 'w', encoding='utf-8') as f:
                await f.write(final_content)
            return "Patched successfully (Fuzzy match: ignored whitespace/indentation differences)."

        # --- 失败：提供详细诊断信息 ---
        # 帮助 AI 找到它可能想改的地方
        first_line = old_lines[0].strip()[:50]
        candidates = []
        for i, line in enumerate(lines):
            if first_line in line.strip():
                candidates.append(f"Line {i+1}: {line.strip()[:80]}")
        
        error_msg = f"[Error] old_string not found in '{path}'.\n"
        error_msg += "Check line endings or indentation.\n"
        if candidates:
            error_msg += "Did you mean one of these locations?\n" + "\n".join(candidates[:3])
            
        return error_msg

    except Exception as e:
        return f"[Error] Patch failed: {str(e)}"

async def todo_write_tool_local(action: str, id: str = None, content: str = None, priority: str = "medium", status: str = None) -> str:
    """本地环境任务管理"""
    try:
        # 1. 获取当前工作目录并确保 .agent 文件夹存在
        cwd = await _get_current_cwd()
        party_dir = Path(cwd) / ".agent"
        if not party_dir.exists():
            await aiofiles.os.makedirs(party_dir, exist_ok=True)
        
        todo_file = party_dir / "ai_todos.json"
        
        # 2. 读取现有数据
        todos = []
        if todo_file.exists():
            try:
                async with aiofiles.open(todo_file, 'r', encoding='utf-8') as f:
                    file_content = await f.read()
                    if file_content.strip():
                        todos = json.loads(file_content)
            except (json.JSONDecodeError, Exception) as e:
                print(f"读取 Todo 文件失败，将初始化为空列表: {e}")
                todos = []
            
        msg = ""

        # 3. 执行逻辑操作
        if action == "create":
            if not content: 
                return "[Error] Content required for creation."
            new_todo = {
                "id": id or str(uuid.uuid4())[:8],
                "content": content,
                "priority": priority,
                "status": "pending",
                "created_at": datetime.now().isoformat()
            }
            todos.append(new_todo)
            msg = f"[Success] Created local todo: {new_todo['id']}"
            
        elif action == "list":
            if not todos: 
                return "No todos found in this project."
            lines = ["📋 **Project Todos (Local)**:"]
            # 排序：未完成的在前，已完成的在后
            sorted_todos = sorted(todos, key=lambda x: x.get('status') == 'done')
            for t in sorted_todos:
                status_icon = "✅" if t.get('status') == 'done' else "⏳"
                priority_map = {"high": "🔴", "medium": "🟡", "low": "🟢"}
                p_icon = priority_map.get(t.get('priority', 'medium'), "⚪")
                lines.append(f"{status_icon} {p_icon} [{t['id']}] {t['content'][:50]}")
            return "\n".join(lines)

        elif action in ["update", "toggle", "delete"]:
            if not id: 
                return "[Error] ID required for update/toggle/delete."
            
            target = next((t for t in todos if t['id'] == id), None)
            if not target: 
                return f"[Error] ID {id} not found."
            
            if action == "delete":
                todos.remove(target)
                msg = f"[Success] Deleted local todo: {id}"

            elif action == "toggle":
                # 切换逻辑
                if target.get('status') != 'done':
                    target['status'] = 'done'
                    target['completed_at'] = datetime.now().isoformat() # 记录完成时间
                else:
                    target['status'] = 'pending'
                    target['completed_at'] = None # 重置完成时间
                msg = f"[Success] Toggled local todo {id} to {target['status']}"

            elif action == "update":
                if content: target['content'] = content
                if priority: target['priority'] = priority
                
                # 处理状态更新和完成时间
                if status:
                    # 如果状态从非 done 变为 done
                    if status == "done" and target.get('status') != "done":
                        target['completed_at'] = datetime.now().isoformat()
                    # 如果状态从 done 变为非 done
                    elif status != "done" and target.get('status') == "done":
                        target['completed_at'] = None
                    
                    target['status'] = status
                
                target['updated_at'] = datetime.now().isoformat()
                msg = f"[Success] Updated local todo: {id}"
        else:
            return f"[Error] Unknown action: {action}"

        # 4. 异步写回本地文件
        async with aiofiles.open(todo_file, 'w', encoding='utf-8') as f:
            await f.write(json.dumps(todos, indent=2, ensure_ascii=False))
            
        return msg

    except Exception as e:
        return f"[Error] Local Todo operation failed: {str(e)}"
# ==================== Claude & Qwen Agents (恢复) ====================

cli_info = "这是一个交互式命令行工具..."

async def claude_code_async(prompt) -> str | AsyncIterator[str]:
    settings = await load_settings()
    cwd = settings.get("CLISettings", {}).get("cc_path")
    ccSettings = settings.get("ccSettings", {})
    if not cwd: return "No working directory."
    
    extra_config = {}
    if ccSettings.get("enabled"):
        extra_config = {
            "ANTHROPIC_BASE_URL": ccSettings.get("base_url"),
            "ANTHROPIC_API_KEY": ccSettings.get("api_key"),
            "ANTHROPIC_MODEL": ccSettings.get("model"),
        }
        extra_config = {k: str(v) if v else "" for k, v in extra_config.items()}

    async def _stream():
        permission_mode=ccSettings.get("permissionMode", "default")
        if permission_mode == "cowork":
            permission_mode = "bypassPermissions"
        options = ClaudeAgentOptions(
            cwd=cwd,
            continue_conversation=True,
            permission_mode=permission_mode,
            env={**os.environ, **extra_config}
        )
        async for message in query(prompt=prompt, options=options):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock): yield block.text
    return _stream()

async def qwen_code_async(prompt: str) -> str | AsyncIterator[str]:
    settings = await load_settings()
    cwd = settings.get("CLISettings", {}).get("cc_path")
    qcSettings = settings.get("qcSettings", {})
    if not cwd: return "No working directory."

    extra_config = {}
    if qcSettings.get("enabled"):
        extra_config = {
            "OPENAI_BASE_URL": str(qcSettings.get("base_url") or ""),
            "OPENAI_API_KEY": str(qcSettings.get("api_key") or ""),
            "OPENAI_MODEL": str(qcSettings.get("model") or ""),
        }
    executable = shutil.which("qwen") or "qwen"

    async def _stream():
        try:
            permission_mode=qcSettings.get("permissionMode", "default")
            if permission_mode == "cowork":
                permission_mode = "yolo"
            process = await asyncio.create_subprocess_exec(
                executable, "-p", prompt, "--approval-mode", permission_mode,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                cwd=cwd, env={**os.environ, **extra_config}
            )
            async for out in _merge_streams(read_stream(process.stdout), read_stream(process.stderr, is_error=True)):
                yield out
            await process.wait()
        except Exception as e: yield str(e)
    return _stream()


# ==================== [新增] Skill 专用读取工具 ====================

async def read_skill_tool_logic(cwd: str, skill_id: str, is_docker: bool = True) -> str:
    """
    内部通用逻辑：读取 Skill 文件夹结构和说明文档。
    若工作区不存在该技能，且全局技能目录可用，则自动复制到工作区（Docker/Local 均支持）。
    """
    skill_rel_path = f".agent/skills/{skill_id}"
    workspace_skill_path = f"/workspace/.agent/skills/{skill_id}" if is_docker else str(Path(cwd) / ".agent" / "skills" / skill_id)

    # ----- 复制逻辑：工作区缺失时，从全局复制 -----
    if is_docker:
        # Docker 环境：利用已映射的全局技能目录
        container_name = await get_or_create_docker_sandbox(cwd)          # 获取/创建容器
        global_skill_path = f"/home/agent/.agents/skills/{skill_id}"      # 容器内全局技能路径
        try:
            # 1. 检查工作区技能是否存在
            test_cmd = ["test", "-d", workspace_skill_path]
            await _exec_docker_cmd_simple(cwd, test_cmd)                  # 不存在会抛出异常
        except Exception:
            # 2. 工作区不存在，尝试从全局复制
            try:
                # 检查全局技能是否存在
                test_global = ["test", "-d", global_skill_path]
                await _exec_docker_cmd_simple(cwd, test_global)

                # 确保目标父目录存在
                mkdir_cmd = ["mkdir", "-p", f"/workspace/.agent/skills"]
                await _exec_docker_cmd_simple(cwd, mkdir_cmd)

                # 执行复制
                cp_cmd = ["cp", "-r", global_skill_path, f"/workspace/.agent/skills/"]
                await _exec_docker_cmd_simple(cwd, cp_cmd)

                print(f"[Skill AutoCopy][Docker] Copied global skill '{skill_id}' to workspace.")
            except Exception as e:
                # 复制失败或全局技能不存在，继续尝试读取工作区（若不存在则后续报错）
                pass
    else:
        # Local 环境：使用 shutil 复制（已实现，但整合到 logic 中统一管理）
        workspace_path = Path(cwd) / ".agent" / "skills" / skill_id
        if not workspace_path.exists():
            global_path = Path(SKILLS_DIR) / skill_id
            if global_path.exists() and global_path.is_dir():
                try:
                    workspace_path.parent.mkdir(parents=True, exist_ok=True)
                    await asyncio.to_thread(
                        shutil.copytree,
                        global_path,
                        workspace_path,
                        dirs_exist_ok=True
                    )
                    print(f"[Skill AutoCopy][Local] Copied global skill '{skill_id}' to workspace.")
                except Exception as e:
                    print(f"[Skill AutoCopy][Local] Copy failed: {e}. Will fallback to global read.")
                    # 降级读取已由主流程处理

    # ----- 原有读取逻辑保持不变（读取工作区技能）-----
    tree_str = ""
    doc_content = ""

    if is_docker:
        try:
            tree_str = await _exec_docker_cmd_simple(cwd, ["find", skill_rel_path, "-maxdepth", "2", "-not", "-path", '*/.*'])
            for name in ["SKILL.md", "skill.md", "SKILLS.md", "skills.md"]:
                try:
                    doc_path = f"{skill_rel_path}/{name}"
                    doc_content = await _exec_docker_cmd_simple(cwd, ["cat", doc_path])
                    break
                except:
                    continue
        except Exception as e:
            return f"[Error] Skill '{skill_id}' not found or inaccessible in Docker: {str(e)}"
    else:
        try:
            base_path = Path(cwd) / ".agent" / "skills" / skill_id
            if not base_path.exists():
                return f"[Error] Skill '{skill_id}' folder does not exist in workspace and auto-copy failed or global skill unavailable."

            # 生成本地文件树（深度 ≤2）
            tree_lines = [f"{skill_id}/"]
            for p in base_path.rglob("*"):
                if p.name.startswith("."): continue
                depth = len(p.relative_to(base_path).parts)
                if depth > 2: continue
                indent = "  " * depth
                tree_lines.append(f"{indent}{p.name}{'/' if p.is_dir() else ''}")
            tree_str = "\n".join(tree_lines)

            # 读取本地说明文档
            for name in ["SKILL.md", "skill.md", "SKILLS.md", "skills.md"]:
                doc_path = base_path / name
                if doc_path.exists():
                    async with aiofiles.open(doc_path, 'r', encoding='utf-8', errors='replace') as f:
                        doc_content = await f.read()
                    break
        except Exception as e:
            return f"[Error] Skill '{skill_id}' read failed: {str(e)}"

    if not doc_content and not tree_str:
        return f"[Error] Could not find skill details for '{skill_id}'."

    res = f"--- Skill Details: {skill_id} ---\n"
    res += f"\n📂 **Folder Structure:**\n```\n{tree_str}\n```\n"
    res += f"\n📖 **Documentation ({skill_rel_path}):**\n\n{doc_content or '(No SKILL.md found)'}"
    return res

async def read_skill_tool(skill_id: str) -> str:
    """[Docker] 读取特定技能的完整文档和文件树"""
    cwd = await _get_current_cwd()
    return await read_skill_tool_logic(cwd, skill_id, is_docker=True)

async def read_skill_tool_local(skill_id: str) -> str:
    """[Local] 读取特定技能的完整文档和文件树"""
    cwd = await _get_current_cwd()
    return await read_skill_tool_logic(cwd, skill_id, is_docker=False)

# ==================== 工具注册表 (完整) ====================

TOOLS_REGISTRY = {
    # --- 只读 ---
    "list_files": {
        "type": "function", "function": {
            "name": "list_files_tool", 
            "description": "List files in docker workspace.",
            "parameters": {
                "type": "object", 
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path to list files in (from workspace root)."
                    }, 
                    "show_all": {"type": "boolean", "default": True}
                }, 
                "required": ["path"]
            }
        }
    },
    "read_file": {
        "type": "function", "function": {
            "name": "read_file_tool", 
            "description": "Read file content.",
            "parameters": {
                "type": "object", 
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path to file (from workspace root)."
                    }
                }, 
                "required": ["path"]
            }
        }
    },
    "read_file_range": {
        "type": "function", "function": {
            "name": "read_file_range_tool", 
            "description": "Read a specific range of lines from a file. Useful for large files after grepping.",
            "parameters": {
                "type": "object", 
                "properties": {
                    "path": {"type": "string", "description": "Relative path to file"},
                    "start_line": {"type": "integer"},
                    "end_line": {"type": "integer"}
                }, 
                "required": ["path", "start_line", "end_line"]
            }
        }
    },
    "tail_file": {
        "type": "function", "function": {
            "name": "tail_file_tool", 
            "description": "Read the last N lines of a file. Useful for reading logs.",
            "parameters": {
                "type": "object", 
                "properties": {
                    "path": {"type": "string", "description": "Relative path to file"},
                    "lines": {"type": "integer", "default": 100, "description": "Number of lines to read from the end"}
                }, 
                "required": ["path"]
            }
        }
    },
    "search_files": {
        "type": "function", "function": {
            "name": "search_files_tool", 
            "description": "Grep search.",
            "parameters": {
                "type": "object", 
                "properties": {
                    "pattern": {"type": "string"}, 
                    "path": {
                        "type": "string",
                        "description": "Relative path to directory to search in (from workspace root)."
                    }
                }, 
                "required": ["pattern"]
            }
        }
    },
    "glob_files": {
        "type": "function", "function": {
            "name": "glob_files_tool", 
            "description": "Recursive glob.",
            "parameters": {
                "type": "object", 
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern (relative to workspace root)."
                    }, 
                    "exclude": {"type": "string"}
                }, 
                "required": ["pattern"]
            }
        }
    },
    "read_skill": {
        "type": "function", "function": {
            "name": "read_skill_tool", 
            "description": "Read full documentation and file tree for a project-specific skill from .agent/skills/.",
            "parameters": {
                "type": "object", 
                "properties": {
                    "skill_id": {"type": "string"}
                }, 
                "required": ["skill_id"]
            }
        }
    },
    # --- 编辑 ---
    "edit_file": {
        "type": "function", "function": {
            "name": "edit_file_tool", 
            "description": "Overwrite file.",
            "parameters": {
                "type": "object", 
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path to file (from workspace root)."
                    }, 
                    "content": {"type": "string"}
                }, 
                "required": ["path", "content"]
            }
        }
    },
    "edit_file_patch": {
        "type": "function", "function": {
            "name": "edit_file_patch_tool", 
            "description": "Precise replacement.",
            "parameters": {
                "type": "object", 
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path to file (from workspace root)."
                    }, 
                    "old_string": {"type": "string"}, 
                    "new_string": {"type": "string"}
                }, 
                "required": ["path", "old_string"]
            }
        }
    },
    # --- 任务 ---
    "todo_write": {
        "type": "function", "function": {
            "name": "todo_write_tool", 
            "description": "Manage tasks.",
            "parameters": {
                "type": "object", 
                "properties": {
                    "action": {"type": "string", "enum": ["create","list","update","delete","toggle"]}, 
                    "content": {"type": "string"}, 
                    "id": {"type": "string"}
                }, 
                "required": ["action"]
            }
        }
    },
    # --- 基础设施 ---
    "bash": {
        "type": "function", "function": {
            "name": "docker_sandbox_async", 
            "description": "Run bash in Docker.",
            "parameters": {
                "type": "object", 
                "properties": {
                    "command": {"type": "string"}, 
                    "background": {"type": "boolean", "description": "Run non-blocking (server/watcher). Returns PID."}
                }, 
                "required": ["command"]
            }
        }
    },
    "manage_processes": {
        "type": "function", "function": {
            "name": "manage_processes_tool", 
            "description": "Check logs or kill background processes (Docker & Local).",
            "parameters": {
                "type": "object", 
                "properties": {
                    "action": {"type": "string", "enum": ["list", "logs", "kill"]},
                    "pid": {"type": "string"}
                }, 
                "required": ["action"]
            }
        }
    },
    "manage_ports": {
        "type": "function", "function": {
            "name": "docker_manage_ports_tool", 
            "description": "Forward Docker ports to localhost.",
            "parameters": {
                "type": "object", 
                "properties": {
                    "action": {"type": "string", "enum": ["forward", "stop", "list"]},
                    "container_port": {"type": "integer"},
                    "host_port": {"type": "integer"}
                }, 
                "required": ["action"]
            }
        }
    }
}

LOCAL_TOOLS_REGISTRY = {
    # --- 只读 ---
    "list_files_local": {
        "type": "function", "function": {
            "name": "list_files_tool_local", 
            "description": "List local files.",
            "parameters": {
                "type": "object", 
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path to list files in (from current working directory)."
                    }, 
                    "show_all": {"type": "boolean","default": True}
                }, 
                "required": ["path"]
            }
        }
    },
    "read_file_local": {
        "type": "function", "function": {
            "name": "read_file_tool_local", 
            "description": "Read local file.",
            "parameters": {
                "type": "object", 
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path to file (from current working directory)."
                    }
                }, 
                "required": ["path"]
            }
        }
    },
    "read_file_range_local": {
        "type": "function", "function": {
            "name": "read_file_range_tool_local", 
            "description": "Read a specific range of lines from a local file. Useful for large files after grepping.",
            "parameters": {
                "type": "object", 
                "properties": {
                    "path": {"type": "string", "description": "Relative path to file"},
                    "start_line": {"type": "integer"},
                    "end_line": {"type": "integer"}
                }, 
                "required": ["path", "start_line", "end_line"]
            }
        }
    },
    "tail_file_local": {
        "type": "function", "function": {
            "name": "tail_file_tool_local", 
            "description": "Read the last N lines of a local file. Useful for reading logs.",
            "parameters": {
                "type": "object", 
                "properties": {
                    "path": {"type": "string", "description": "Relative path to file"},
                    "lines": {"type": "integer", "default": 100}
                }, 
                "required": ["path"]
            }
        }
    },
    "search_files_local": {
         "type": "function", "function": {
            "name": "search_files_tool_local", 
            "description": "Search local files.",
            "parameters": {
                "type": "object", 
                "properties": {
                    "pattern": {"type": "string"}
                    # 注意：根据之前的代码实现，search_files_local 似乎没有 path 参数，而是直接在 CWD 搜索。
                    # 如果需要支持指定路径，需要在实现代码中确认。
                }, 
                "required": ["pattern"]
            }
        }
    },
    "glob_files_local": {
         "type": "function", "function": {
            "name": "glob_files_tool_local", 
            "description": "Glob local files.",
            "parameters": {
                "type": "object", 
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern (relative to current working directory)."
                    }
                }, 
                "required": ["pattern"]
            }
        }
    },
    "read_skill_local": {
        "type": "function", "function": {
            "name": "read_skill_tool_local", 
            "description": "Read full documentation and file tree for a project-specific skill from .agent/skills/ (Local).",
            "parameters": {
                "type": "object", 
                "properties": {
                    "skill_id": {"type": "string"}
                }, 
                "required": ["skill_id"]
            }
        }
    },
    # --- 编辑 ---
    "edit_file_local": {
        "type": "function", "function": {
            "name": "edit_file_tool_local", 
            "description": "Write local file.",
            "parameters": {
                "type": "object", 
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path to file (from current working directory)."
                    }, 
                    "content": {"type": "string"}
                }, 
                "required": ["path"]
            }
        }
    },
    "edit_file_patch_local": {
        "type": "function", "function": {
            "name": "edit_file_patch_tool_local", 
            "description": "Patch local file.",
            "parameters": {
                "type": "object", 
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path to file (from current working directory)."
                    }, 
                    "old_string": {"type": "string"}, 
                    "new_string": {"type": "string"}
                }, 
                "required": ["path", "old_string"]
            }
        }
    },
    "todo_write_local": {
        "type": "function", "function": {
            "name": "todo_write_tool_local", 
            "description": "Manage local tasks.",
            "parameters": {
                "type": "object", 
                "properties": {
                    "action": {"type": "string", "enum": ["create","list","update","delete","toggle"]}, 
                    "content": {"type": "string"}, 
                    "id": {"type": "string"}
                }, 
                "required": ["action"]
            }
        }
    },
    # --- 基础设施 ---
    "bash_local": {
        "type": "function", "function": {
            "name": "bash_tool_local", 
            "description": "Run local command.",
            "parameters": {
                "type": "object", 
                "properties": {
                    "command": {"type": "string"},
                    "background": {"type": "boolean", "description": "Run in background."}
                }, 
                "required": ["command"]
            }
        }
    },
    "manage_processes_local": {
        "type": "function", "function": {
            "name": "manage_processes_tool", 
            "description": "Manage local background processes.",
            "parameters": {
                "type": "object", 
                "properties": {
                    "action": {"type": "string", "enum": ["list", "logs", "kill"]},
                    "pid": {"type": "string"}
                }, 
                "required": ["action"]
            }
        }
    },
    "local_net_tool": {
        "type": "function", "function": {
            "name": "local_net_tool", 
            "description": "Check local ports.",
            "parameters": {
                "type": "object", 
                "properties": {
                    "action": {"type": "string", "enum": ["check", "scan"]},
                    "port": {"type": "integer"}
                }, 
                "required": ["action"]
            }
        }
    }
}

# 代理工具定义 (用于其他Agent)
claude_code_tool = {
    "type": "function",
    "function": {
        "name": "claude_code_async",
        "description": f"Interact with Claude Code Agent. {cli_info}",
        "parameters": {"type": "object", "properties": {"prompt": {"type": "string"}}, "required": ["prompt"]}
    }
}
qwen_code_tool = {
    "type": "function",
    "function": {
        "name": "qwen_code_async",
        "description": f"Interact with Qwen Code Agent. {cli_info}",
        "parameters": {"type": "object", "properties": {"prompt": {"type": "string"}}, "required": ["prompt"]}
    }
}

def get_tools_for_mode(mode: str) -> list:
    """获取 Docker 环境工具集"""
    # 基础只读
    read = [TOOLS_REGISTRY["list_files"], 
            TOOLS_REGISTRY["read_file"], 
            TOOLS_REGISTRY["read_file_range"],
            TOOLS_REGISTRY["tail_file"],     
            TOOLS_REGISTRY["search_files"], 
            TOOLS_REGISTRY["glob_files"],
            TOOLS_REGISTRY["read_skill"]
            ]
    # 编辑
    edit = [TOOLS_REGISTRY["edit_file"], TOOLS_REGISTRY["edit_file_patch"], TOOLS_REGISTRY["todo_write"]]
    # 基础设施 (执行/进程/端口)
    infra = [TOOLS_REGISTRY["bash"], TOOLS_REGISTRY["manage_processes"], TOOLS_REGISTRY["manage_ports"]]
    
    if mode == "default": return read
    if mode == "auto-approve": return read + edit + [TOOLS_REGISTRY["manage_processes"]]
    if mode == "yolo": return read + edit + infra
    return read

def get_local_tools_for_mode(mode: str) -> list:
    """获取 Local 环境工具集"""
    read = [
        LOCAL_TOOLS_REGISTRY["list_files_local"], 
        LOCAL_TOOLS_REGISTRY["read_file_local"], 
        LOCAL_TOOLS_REGISTRY["read_file_range_local"],
        LOCAL_TOOLS_REGISTRY["tail_file_local"],    
        LOCAL_TOOLS_REGISTRY["search_files_local"], 
        LOCAL_TOOLS_REGISTRY["glob_files_local"],
        LOCAL_TOOLS_REGISTRY["read_skill_local"] 
    ]
    edit = [LOCAL_TOOLS_REGISTRY["edit_file_local"], LOCAL_TOOLS_REGISTRY["edit_file_patch_local"], LOCAL_TOOLS_REGISTRY["todo_write_local"]]
    infra = [
        LOCAL_TOOLS_REGISTRY["bash_local"], 
        LOCAL_TOOLS_REGISTRY["manage_processes_local"],
        LOCAL_TOOLS_REGISTRY["local_net_tool"]
    ]
    
    if mode == "default": return read
    if mode == "auto-approve": return read + edit + [LOCAL_TOOLS_REGISTRY["manage_processes_local"], LOCAL_TOOLS_REGISTRY["local_net_tool"]]
    if mode == "yolo": return read + edit + infra
    return read