# py/skills.py

import shutil
import tempfile
import json
import os
import httpx
from pathlib import Path
from urllib.parse import urlparse
from fastapi import APIRouter, HTTPException, UploadFile, File, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional
import asyncio

from py.get_setting import SKILLS_DIR

router = APIRouter(prefix="/api/skills", tags=["skills"])

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

# ==================== 工具函数 ====================

def robust_rmtree(path: Path):
    """强制删除目录"""
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)

import yaml
import re
from pathlib import Path

def get_skill_metadata(skill_dir: Path, skill_id: str) -> Skill:
    """
    根据 Agent Skills 标准解析元数据。
    规范：查找 SKILL.md 或 skills.md 顶部的 YAML Frontmatter。
    """
    # 按照规范优先级查找元数据文件
    target_files = ["SKILL.md", "skill.md","SKILLS.md", "skills.md"]
    meta_file = None
    for filename in target_files:
        p = skill_dir / filename
        if p.exists():
            meta_file = p
            break
    
    meta = {}
    if meta_file:
        try:
            content = meta_file.read_text(encoding="utf-8")
            # 使用正则提取两个 --- 之间的 YAML 部分
            # 支持 Windows (\r\n) 和 Unix (\n) 换行符
            match = re.search(r'^---\s*\n(.*?)\n---\s*', content, re.DOTALL | re.MULTILINE)
            if match:
                yaml_text = match.group(1)
                # 使用 yaml.safe_load 解析文本
                parsed_meta = yaml.safe_load(yaml_text)
                if isinstance(parsed_meta, dict):
                    meta = parsed_meta
        except Exception as e:
            print(f"解析 {meta_file.name} 元数据失败: {e}")

    # 获取文件列表供前端展示（排除隐藏文件）
    file_list = [f.name for f in skill_dir.iterdir() if f.is_file() and not f.name.startswith('.')]
    
    # 返回 Skill 对象
    # metadata 字段通常包含作者、版本等，如果 YAML 里有则优先提取
    return Skill(
        id=skill_id,
        name=meta.get("name", skill_id),
        description=meta.get("description", "Agent 智能体技能"),
        version=str(meta.get("version", "1.0.0")),
        author=meta.get("author") or meta.get("metadata", {}).get("author", "Local"),
        files=file_list[:8]  # 展示前8个文件
    )

@router.get("/{skill_id}/content")
async def get_skill_content(skill_id: str):
    """读取指定技能的 SKILL.md 全文内容"""
    skill_dir = Path(SKILLS_DIR) / skill_id
    if not skill_dir.exists():
        raise HTTPException(status_code=404, detail="技能不存在")
    
    target_files = ["SKILL.md", "skill.md","SKILLS.md", "skills.md"]
    for filename in target_files:
        p = skill_dir / filename
        if p.exists():
            try:
                content = p.read_text(encoding="utf-8")
                return {"content": content}
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"读取文件失败: {e}")
                
    raise HTTPException(status_code=404, detail="未找到元数据文件 (SKILL.md)")

def github_url_to_zip(url: str) -> str:
    """转换 GitHub URL 为 ZIP 下载链接"""
    url = url.strip().rstrip('/').removesuffix('.git')
    parsed = urlparse(url)
    path_parts = parsed.path.strip('/').split('/')
    
    if len(path_parts) < 2:
        raise ValueError("无效的 GitHub URL")
        
    owner, repo = path_parts[0], path_parts[1]
    # 默认为 main 分支，如果以后需要支持其他分支可以扩展
    return f"https://github.com/{owner}/{repo}/archive/refs/heads/main.zip"

async def download_zip(url: str, dest: Path):
    async with httpx.AsyncClient(follow_redirects=True) as client:
        async with client.stream("GET", url) as resp:
            if resp.status_code != 200:
                raise Exception(f"下载失败: Status {resp.status_code}")
            with open(dest, "wb") as f:
                async for chunk in resp.aiter_bytes():
                    f.write(chunk)

# ==================== 核心逻辑 ====================

def _process_github_install(url: str):
    """后台任务：处理 GitHub 安装逻辑"""
    temp_dir = Path(tempfile.mkdtemp())
    try:
        zip_url = github_url_to_zip(url)
        zip_path = temp_dir / "repo.zip"
        
        # 1. 下载
        asyncio.run(download_zip(zip_url, zip_path))
        
        # 2. 解压
        extract_dir = temp_dir / "extracted"
        shutil.unpack_archive(zip_path, extract_dir)
        
        # 3. 寻找内容根目录 (通常是 repo-main)
        root_content = next(extract_dir.iterdir())
        if not root_content.is_dir():
            root_content = extract_dir
            
        # 4. 寻找 skills 文件夹
        skills_source_dir = root_content / "skills"
        
        if not skills_source_dir.exists():
            # 如果没有 skills 文件夹，尝试直接把整个仓库当作一个 skill 导入
            # 这种情况比较少见，但为了健壮性
            print(f"未找到 skills 文件夹，尝试将根目录视为单个技能导入: {root_content.name}")
            dest_path = Path(SKILLS_DIR) / root_content.name
            robust_rmtree(dest_path)
            shutil.copytree(root_content, dest_path)
            return

        # 5. 遍历 skills 文件夹下的子目录并导入
        imported_count = 0
        for item in skills_source_dir.iterdir():
            if item.is_dir():
                dest_path = Path(SKILLS_DIR) / item.name
                # 如果存在则覆盖
                robust_rmtree(dest_path)
                shutil.copytree(item, dest_path)
                imported_count += 1
                
        print(f"从 {url} 成功导入了 {imported_count} 个技能")

    except Exception as e:
        print(f"GitHub Skill 安装失败: {str(e)}")
    finally:
        robust_rmtree(temp_dir)

# ==================== API 路由 ====================

@router.get("/list", response_model=SkillsResponse)
async def list_skills():
    """列出已安装的 Skills"""
    if not os.path.exists(SKILLS_DIR):
        os.makedirs(SKILLS_DIR, exist_ok=True)
        return SkillsResponse(skills=[])
    
    skills_list = []
    base = Path(SKILLS_DIR)
    
    for item in base.iterdir():
        if item.is_dir():
            skills_list.append(get_skill_metadata(item, item.name))
            
    return SkillsResponse(skills=skills_list)

@router.delete("/{skill_id}")
async def delete_skill(skill_id: str):
    """删除 Skill"""
    target = Path(SKILLS_DIR) / skill_id
    if target.exists():
        robust_rmtree(target)
    return {"status": "success"}

@router.post("/install-from-github")
async def install_skill_github(req: GitHubSkillInstallRequest, bg_tasks: BackgroundTasks):
    """从 GitHub 安装 (扫描 skills 目录)"""
    bg_tasks.add_task(_process_github_install, req.url)
    return {"status": "processing", "message": "后台安装任务已启动"}

@router.post("/upload-zip")
async def upload_skill_zip(file: UploadFile = File(...)):
    """上传本地 ZIP (单技能模式)"""
    if not file.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="仅支持 zip 文件")

    # 假设文件名即为技能 ID (去除 .zip)
    skill_id = Path(file.filename).stem
    target_dir = Path(SKILLS_DIR) / skill_id
    
    with tempfile.TemporaryDirectory() as td:
        temp_path = Path(td)
        zip_file = temp_path / "upload.zip"
        
        # 保存上传文件
        with open(zip_file, "wb") as f:
            shutil.copyfileobj(file.file, f)
            
        # 解压
        extract_dir = temp_path / "extracted"
        shutil.unpack_archive(zip_file, extract_dir)
        
        # 处理解压后的层级结构
        # 如果解压后只有一个文件夹，且该文件夹内包含内容，则剥离这一层
        items = list(extract_dir.iterdir())
        if len(items) == 1 and items[0].is_dir():
            source = items[0]
        else:
            source = extract_dir

        # 安装/覆盖
        robust_rmtree(target_dir)
        shutil.copytree(source, target_dir)
        
    return {"status": "success", "skill_id": skill_id}