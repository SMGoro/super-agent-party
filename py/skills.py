# py/skills.py

import shutil
import tempfile
import json
import os
import httpx
import yaml
import re
import asyncio
from pathlib import Path
from urllib.parse import urlparse
from fastapi import APIRouter, HTTPException, UploadFile, File, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional

from py.get_setting import SKILLS_DIR

router = APIRouter(prefix="/api/skills", tags=["skills"])

# ==================== 数据模型 ====================

class Skill(BaseModel):
    id: str
    name: str
    description: str = "暂无描述"
    version: str = "1.0.0"
    author: str = "未知"
    files: List[str] = []

class SkillsResponse(BaseModel):
    skills: List[Skill]

class GitHubSkillInstallRequest(BaseModel):
    url: str

class SkillSyncRequest(BaseModel):
    skill_id: str
    project_path: str
    action: str  # "install" 或 "remove"

# ==================== 工具函数 ====================

def robust_rmtree(path: Path):
    """强制删除目录，处理 Windows 权限或被占用问题"""
    if path.exists():
        try:
            shutil.rmtree(path, ignore_errors=True)
        except Exception as e:
            print(f"删除目录 {path} 失败: {e}")

def parse_github_url(url: str):
    """
    解析 GitHub URL，支持深度链接。
    例如: https://github.com/anthropics/skills/tree/main/skills/docx
    返回: (zip_download_url, branch, subpath)
    """
    url = url.strip().rstrip('/').removesuffix('.git')
    # 正则匹配 owner, repo 和可能的 tree/branch/path
    pattern = r"github\.com/([^/]+)/([^/]+)(?:/(?:tree|blob)/([^/]+)/(.*))?"
    match = re.search(pattern, url)
    
    if not match:
        raise ValueError("无效的 GitHub URL")
        
    owner, repo, branch, subpath = match.groups()
    branch = branch or "main" 
    
    zip_url = f"https://github.com/{owner}/{repo}/archive/refs/heads/{branch}.zip"
    return zip_url, branch, subpath

async def download_zip(url: str, dest: Path):
    """异步下载文件"""
    async with httpx.AsyncClient(follow_redirects=True, timeout=60.0) as client:
        async with client.stream("GET", url) as resp:
            if resp.status_code != 200:
                raise Exception(f"下载失败: Status {resp.status_code}")
            with open(dest, "wb") as f:
                async for chunk in resp.aiter_bytes():
                    f.write(chunk)

def get_skill_metadata(skill_dir: Path, skill_id: str) -> Skill:
    """解析技能元数据 (SKILL.md 的 YAML Frontmatter)"""
    target_files = ["SKILL.md", "skill.md", "SKILLS.md", "skills.md"]
    meta_file = next((skill_dir / f for f in target_files if (skill_dir / f).exists()), None)
    
    meta = {}
    if meta_file:
        try:
            content = meta_file.read_text(encoding="utf-8")
            # 提取 --- 之间的 YAML
            match = re.search(r'^---\s*\n(.*?)\n---\s*', content, re.DOTALL | re.MULTILINE)
            if match:
                yaml_text = match.group(1)
                parsed_meta = yaml.safe_load(yaml_text)
                if isinstance(parsed_meta, dict):
                    meta = parsed_meta
        except Exception as e:
            print(f"解析 {meta_file.name} 失败: {e}")

    # 获取文件列表
    file_list = [f.name for f in skill_dir.iterdir() if f.is_file() and not f.name.startswith('.')]
    
    return Skill(
        id=skill_id,
        name=meta.get("name", skill_id),
        description=meta.get("description", "Agent 智能体技能"),
        version=str(meta.get("version", "1.0.0")),
        author=meta.get("author") or meta.get("metadata", {}).get("author", "Local"),
        files=file_list[:8]
    )

# ==================== 核心安装逻辑 ====================

def _install_skills_from_directory(source_dir: Path) -> List[str]:
    """
    智能安装处理器：
    1. 如果 source_dir 包含 SKILL.md，视为单技能安装。
    2. 否则，检查是否包含 skills/ 目录。
    3. 否则，扫描所有子目录，安装包含 SKILL.md 的子目录。
    """
    installed_ids = []
    target_files = ["SKILL.md", "skill.md", "SKILLS.md", "skills.md"]

    def is_skill_dir(d: Path):
        return any((d / f).exists() for f in target_files)

    # 1. 检查本身是否就是技能
    if is_skill_dir(source_dir):
        skill_id = source_dir.name
        dest_path = Path(SKILLS_DIR) / skill_id
        robust_rmtree(dest_path)
        shutil.copytree(source_dir, dest_path)
        installed_ids.append(skill_id)
        return installed_ids

    # 2. 检查内部是否有 skills 文件夹
    search_dir = source_dir
    multi_skills_dir = source_dir / "skills"
    if multi_skills_dir.exists() and multi_skills_dir.is_dir():
        search_dir = multi_skills_dir

    # 3. 扫描子目录
    for item in search_dir.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            if is_skill_dir(item):
                dest_path = Path(SKILLS_DIR) / item.name
                robust_rmtree(dest_path)
                shutil.copytree(item, dest_path)
                installed_ids.append(item.name)
    
    return installed_ids

def _process_github_install(url: str):
    """后台任务：解析 -> 下载 -> 智能安装"""
    temp_dir = Path(tempfile.mkdtemp())
    try:
        zip_url, branch, subpath = parse_github_url(url)
        zip_path = temp_dir / "repo.zip"
        
        # 1. 下载
        asyncio.run(download_zip(zip_url, zip_path))
        
        # 2. 解压
        extract_dir = temp_dir / "extracted"
        shutil.unpack_archive(zip_path, extract_dir)
        
        # 3. 定位内容根目录 (GitHub ZIP 第一层通常是 repo-main)
        repo_root = next(extract_dir.iterdir())
        
        # 4. 如果有 subpath，则进到 subpath 里
        target_source = repo_root
        if subpath:
            potential_path = repo_root.joinpath(*subpath.split('/'))
            if potential_path.exists():
                target_source = potential_path
        
        # 5. 调用统一安装器
        ids = _install_skills_from_directory(target_source)
        print(f"GitHub 安装完成: {ids}")

    except Exception as e:
        print(f"GitHub 安装失败: {str(e)}")
    finally:
        robust_rmtree(temp_dir)

# ==================== API 路由 ====================

@router.get("/list", response_model=SkillsResponse)
async def list_skills():
    """列出所有已安装的全局技能"""
    if not os.path.exists(SKILLS_DIR):
        os.makedirs(SKILLS_DIR, exist_ok=True)
        return SkillsResponse(skills=[])
    
    skills_list = []
    base = Path(SKILLS_DIR)
    for item in sorted(base.iterdir()):
        if item.is_dir() and not item.name.startswith('.'):
            skills_list.append(get_skill_metadata(item, item.name))
    return SkillsResponse(skills=skills_list)

@router.get("/{skill_id}/content")
async def get_skill_content(skill_id: str):
    """前端预览：读取 SKILL.md 的全文"""
    skill_dir = Path(SKILLS_DIR) / skill_id
    if not skill_dir.exists():
        raise HTTPException(status_code=404, detail="技能不存在")
    
    target_files = ["SKILL.md", "skill.md", "SKILLS.md", "skills.md"]
    for filename in target_files:
        p = skill_dir / filename
        if p.exists():
            return {"content": p.read_text(encoding="utf-8")}
                
    raise HTTPException(status_code=404, detail="未找到元数据文件 (SKILL.md)")

@router.post("/install-from-github")
async def install_skill_github(req: GitHubSkillInstallRequest, bg_tasks: BackgroundTasks):
    """从 GitHub 安装，支持具体路径或整个仓库"""
    bg_tasks.add_task(_process_github_install, req.url)
    return {"status": "processing", "message": "后台安装任务已启动"}

@router.post("/upload-zip")
async def upload_skill_zip(file: UploadFile = File(...)):
    """本地 ZIP 上传，支持单技能压缩包或多技能仓库压缩包"""
    if not file.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="仅支持 zip 文件")

    with tempfile.TemporaryDirectory() as td:
        temp_path = Path(td)
        zip_file = temp_path / "upload.zip"
        with open(zip_file, "wb") as f:
            shutil.copyfileobj(file.file, f)
            
        extract_dir = temp_path / "extracted"
        shutil.unpack_archive(zip_file, extract_dir)
        
        # 处理可能的“包一层”目录结构
        items = [i for i in extract_dir.iterdir() if not i.name.startswith('.')]
        source = items[0] if len(items) == 1 and items[0].is_dir() else extract_dir

        installed_ids = _install_skills_from_directory(source)
        
    if not installed_ids:
        raise HTTPException(status_code=400, detail="未检测到有效的 Agent Skill 结构")
        
    return {"status": "success", "installed": installed_ids}

@router.delete("/{skill_id}")
async def delete_skill(skill_id: str):
    """从全局存储中删除技能"""
    target = Path(SKILLS_DIR) / skill_id
    if target.exists():
        robust_rmtree(target)
    return {"status": "success"}

@router.get("/project-status")
async def get_project_skills_status(path: str):
    """查询指定项目已开启了哪些技能"""
    if not path or not os.path.exists(path):
        return {"installed_ids": []}
    
    project_skills_dir = Path(path) / ".party" / "skills"
    if not project_skills_dir.exists():
        return {"installed_ids": []}
    
    return {"installed_ids": [item.name for item in project_skills_dir.iterdir() if item.is_dir()]}

@router.post("/sync")
async def sync_skill_to_project(req: SkillSyncRequest):
    """在全局目录和项目目录之间同步技能"""
    if not req.project_path or not os.path.exists(req.project_path):
        raise HTTPException(status_code=400, detail="项目路径无效")

    global_skill_path = Path(SKILLS_DIR) / req.skill_id
    project_skills_dir = Path(req.project_path) / ".party" / "skills"
    target_path = project_skills_dir / req.skill_id

    if req.action == "install":
        if not global_skill_path.exists():
            raise HTTPException(status_code=404, detail="全局技能不存在，请先安装到系统")
        
        project_skills_dir.mkdir(parents=True, exist_ok=True)
        robust_rmtree(target_path)
        shutil.copytree(global_skill_path, target_path)
        return {"status": "success", "message": f"技能 {req.skill_id} 已同步至项目"}

    elif req.action == "remove":
        robust_rmtree(target_path)
        return {"status": "success", "message": f"技能 {req.skill_id} 已从项目移除"}
    
    raise HTTPException(status_code=400, detail="无效的操作类型")

# 在 py/skills.py 中添加

@router.get("/get_path")
async def get_skills_path():
    """获取技能存储目录的绝对路径"""
    try:
        # 确保目录存在
        abs_path = os.path.abspath(SKILLS_DIR)
        if not os.path.exists(abs_path):
            os.makedirs(abs_path, exist_ok=True)
        return {"path": abs_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))