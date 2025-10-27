# extensions.py
import stat
import shutil
import tempfile
import subprocess
from pathlib import Path
from urllib.parse import urlparse
import httpx
from fastapi import APIRouter, HTTPException,BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional
import os
import json
from fastapi import UploadFile, File
from py.get_setting import EXT_DIR
router = APIRouter(prefix="/api/extensions", tags=["extensions"])

class Extension(BaseModel):
    id: str
    name: str
    description: str = "无描述"
    version: str = "1.0.0"
    author: str = "未知"
    systemPrompt: str = ""
    repository: str = ""
    category: str = ""
    transparent: bool = False
    width: int = 800
    height: int = 600

class ExtensionsResponse(BaseModel):
    extensions: List[Extension]

@router.get("/list", response_model=ExtensionsResponse)
async def list_extensions():
    """获取所有可用的扩展列表"""
    try:
        extensions_dir = EXT_DIR
        
        # 确保扩展目录存在
        if not os.path.exists(extensions_dir):
            os.makedirs(extensions_dir, exist_ok=True)
            return ExtensionsResponse(extensions=[])
        
        # 读取扩展目录
        extensions = []
        for dir_name in os.listdir(extensions_dir):
            dir_path = os.path.join(extensions_dir, dir_name)
            if os.path.isdir(dir_path):
                ext_id = dir_name
                index_path = os.path.join(dir_path, "index.html")
                
                # 检查index.html是否存在
                if os.path.exists(index_path):
                    # 尝试读取扩展的package.json获取元数据
                    package_path = os.path.join(dir_path, "package.json")
                    if os.path.exists(package_path):
                        try:
                            with open(package_path, 'r', encoding='utf-8') as f:
                                package_data = json.load(f)
                                
                            extensions.append(Extension(
                                id=ext_id,
                                name=package_data.get("name", ext_id),
                                description=package_data.get("description", "无描述"),
                                version=package_data.get("version", "1.0.0"),
                                author=package_data.get("author", "未知"),
                                systemPrompt = package_data.get("systemPrompt", ""),
                                repository = package_data.get("repository", ""),
                                category = package_data.get("category", ""),
                                transparent = package_data.get("transparent", False),
                                width = package_data.get("width", 800),
                                height = package_data.get("height", 600)
                            ))
                        except json.JSONDecodeError:
                            # package.json解析失败，使用默认值
                            extensions.append(Extension(
                                id=ext_id,
                                name=ext_id
                            ))
                    else:
                        # 没有package.json，使用默认值
                        extensions.append(Extension(
                            id=ext_id,
                            name=ext_id
                        ))
        
        return ExtensionsResponse(extensions=extensions)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取扩展列表失败: {str(e)}")


def _remove_readonly(func, path, exc_info):
    """onexc 回调：如果失败就改权限再删一次"""
    os.chmod(path, stat.S_IWRITE)
    func(path)

def robust_rmtree(target: Path):
    target = Path(target)
    if not target.exists():
        return
    # Python ≥3.12 用 onexc；旧版本用 onerror 即可
    kwargs = {"onexc": _remove_readonly} if hasattr(shutil, "rmtree") and "onexc" in shutil.rmtree.__annotations__ else {"onerror": _remove_readonly}
    shutil.rmtree(target, **kwargs)

# ------------------ 删除扩展 ------------------
@router.delete("/{ext_id}", status_code=204)
async def delete_extension(ext_id: str):
    target = Path(EXT_DIR) / ext_id
    if not target.exists():
        raise HTTPException(status_code=404, detail="扩展不存在")
    try:
        robust_rmtree(target)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除失败: {e}")
    return

# ------------------ GitHub 安装 ------------------
class GitHubInstallRequest(BaseModel):
    url: str          # 支持 https://github.com/owner/repo 或 直接 zip 下载地址

def _run_bg_install(url: str, ext_id: str):
    """后台任务：下载/克隆并解压到 EXT_DIR/ext_id"""
    target = Path(EXT_DIR) / ext_id
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp())

    try:
        if url.endswith(".zip"):
            zip_path = temp_dir / "repo.zip"
            # 下载
            with httpx.stream("GET", url, follow_redirects=True) as resp:
                resp.raise_for_status()
                with open(zip_path, "wb") as f:
                    for chunk in resp.iter_bytes():
                        f.write(chunk)
            # 解压
            shutil.unpack_archive(zip_path, temp_dir)
            # GitHub zip 解压后多一层 repo-name 目录
            inner = next(temp_dir.iterdir())
            if inner.is_dir() and inner.name != "repo.zip":
                shutil.move(str(inner), str(target))
        else:
            # git clone
            subprocess.run(
                ["git", "clone", url, str(temp_dir / "repo")],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            shutil.move(str(temp_dir / "repo"), str(target))
    except Exception as e:
        # 出错时清理半成品
        if target.exists():
            shutil.rmtree(target)
        raise e
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

@router.post("/install-from-github")
async def install_from_github(
    req: GitHubInstallRequest, background: BackgroundTasks
):
    """
    1. 解析仓库名作为 ext_id
    2. 如果已存在直接 409
    3. 后台任务克隆/下载
    4. 立即返回 202，前端轮询或 websocket 通知可后续扩展
    """
    parse = urlparse(req.url)
    if not parse.netloc or "github" not in parse.netloc:
        raise HTTPException(status_code=400, detail="仅支持 GitHub 地址")
    # 取 owner/repo 作为 ext_id
    path_parts = parse.path.strip("/").split("/")
    if len(path_parts) < 2:
        raise HTTPException(status_code=400, detail="URL 格式错误")
    ext_id = f"{path_parts[0]}_{path_parts[1]}"
    target = Path(EXT_DIR) / ext_id
    if target.exists():
        raise HTTPException(status_code=409, detail="扩展已存在")
    # 后台执行
    background.add_task(_run_bg_install, req.url, ext_id)
    return {"ext_id": ext_id, "status": "installing"}

def find_root_dir(temp_path: Path) -> Path:
    """
    如果 zip 解压后只有 1 个一级目录，并且该目录下有 index.html，
    就认为这一层是多余的，返回其子目录；否则返回原路径。
    """
    entries = [p for p in temp_path.iterdir() if p.is_dir()]
    if len(entries) == 1 and (entries[0] / "index.html").exists():
        return entries[0]
    return temp_path

@router.post("/upload-zip")
async def upload_zip(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="仅支持 zip 文件")
    ext_id = Path(file.filename).stem
    target = Path(EXT_DIR) / ext_id
    if target.exists():
        raise HTTPException(status_code=409, detail="扩展已存在")

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        # 1. 保存上传的 zip
        zip_path = tmp / "up.zip"
        with zip_path.open("wb") as f:
            shutil.copyfileobj(file.file, f)
        # 2. 解压到 tmp/unpack
        unpack = tmp / "unpack"
        shutil.unpack_archive(zip_path, unpack)
        # 3. 去掉可能的中间目录
        real_root = find_root_dir(unpack)
        # 4. 移动到最终位置
        target.mkdir(parents=True, exist_ok=True)
        for item in real_root.iterdir():
            shutil.move(str(item), str(target))

    return {"ext_id": ext_id, "status": "ok"}

class RemotePluginItem(BaseModel):
    id : str 
    name: str
    description: str
    author: str
    version: str
    category: str = "Unknown"
    repository: str          # 唯一标识
    installed: bool = False  # 后端自动填充

class RemotePluginList(BaseModel):
    plugins: List[RemotePluginItem]

@router.get("/remote-list", response_model=RemotePluginList)
async def remote_plugin_list():
    try:
        # 1. 拉取原始 JSON
        gh_url = ("https://raw.githubusercontent.com/"
                  "super-agent-party/super-agent-party.github.io/"
                  "main/plugins.json")
        async with httpx.AsyncClient(timeout=10) as cli:
            resp = await cli.get(gh_url)
            resp.raise_for_status()
            remote = resp.json()          # 现在是真正的 List[dict]
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"无法获取远程插件列表: {e}")

    # 2. 本地已安装
    try:
        local_res = await list_extensions()
        installed_repos = {
            ext.repository.strip().rstrip("/").lower()
            for ext in local_res.extensions
            if ext.repository
        }
    except Exception:
        installed_repos = set()

    # 3. 合并状态
    def _with_status(p: dict):
        repo = p.get("repository", "").strip().rstrip("/").lower()
        parse = urlparse(p.get("repository", ""))
        path_parts = parse.path.strip("/").split("/")
        ext_id = f"{path_parts[0]}_{path_parts[1]}"
        return RemotePluginItem(
            id = ext_id,
            name=p.get("name", "未命名"),
            description=p.get("description", ""),
            author=p.get("author", "未知"),
            version=p.get("version", "1.0.0"),
            category=p.get("category", "Unknown"),
            repository=p.get("repository", ""),
            installed=repo in installed_repos,
        )

    return RemotePluginList(plugins=[_with_status(p) for p in remote])