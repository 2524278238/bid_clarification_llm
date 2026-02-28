import os
import json
import requests
import re
import urllib.parse
import hashlib
import asyncio
from typing import List, Optional
from pathlib import Path
from fastapi.responses import JSONResponse
from .extract_bid_items import extract_bid_purification_items
from extract_service.api_server import extract as extract_service
import shutil

# Import logger
import logging
logger = logging.getLogger("bid_server.bid_purification")

# bid_server 根目录（相对路径，可移植）
BID_SERVER_ROOT = Path(__file__).resolve().parent.parent

# 全局 MD 缓存目录
MD_CACHE_DIR = BID_SERVER_ROOT / "out" / "md_cache"
MD_CACHE_DIR.mkdir(parents=True, exist_ok=True)

def _get_env(key: str, default: None) -> str | None:
    v = os.environ.get(key)
    return v if v else default


def _fetch_file_directly_sync(url: str, timeout: int = 300) -> Optional[bytes]:
    """
    直接获取文件内容（同步版本），绕过本地网关避免死锁。
    
    支持的路由：
    - /template/{dataset_id}/{doc_id} -> 直接访问 ragflow API
    - /file/{uid}/{filename} -> 直接读取本地文件
    - 其他 URL -> 使用 requests.get
    
    Returns:
        文件内容的 bytes，失败返回 None
    """
    from urllib.request import Request, urlopen
    
    try:
        parsed = urllib.parse.urlparse(url)
        route_val = None
        
        # 从 query 参数中提取 route
        if parsed.query:
            query_params = urllib.parse.parse_qs(parsed.query)
            route_val = query_params.get("route", [""])[0]
            if route_val:
                route_val = urllib.parse.unquote(route_val)
        
        if route_val:
            # 处理 /template/ 路由 - 直接访问 ragflow
            if route_val.startswith("/template/"):
                parts = route_val.split("/")
                if len(parts) >= 4:
                    dataset_id = parts[2]
                    doc_id = parts[3]
                    ragflow_host = _get_env("RAGFLOW_BASE_HOST", "http://127.0.0.1:8801")
                    ragflow_key = _get_env("RAGFLOW_API_KEY", "")
                    direct_url = f"{ragflow_host.rstrip('/')}/api/v1/datasets/{dataset_id}/documents/{doc_id}"
                    logger.info(f"[清标服务] 直接访问RAGFlow: {url[:60]}... -> {direct_url[:80]}...")
                    req = Request(direct_url, headers={
                        "Authorization": f"Bearer {ragflow_key}",
                        "Connection": "close"
                    })
                    with urlopen(req, timeout=timeout) as resp:
                        return resp.read()
            
            # 处理 /file/ 路由 - 直接读取本地文件
            if route_val.startswith("/file/"):
                parts = route_val.split("/")
                if len(parts) >= 4:
                    file_uid = parts[2]
                    # filename = "/".join(parts[3:])  # 支持文件名中包含 /
                    file_upload_out = _get_env("FILE_UPLOAD_OUT_DIR", str(BID_SERVER_ROOT / "out" / "file_upload"))
                    p_out = Path(file_upload_out)
                    if not p_out.is_absolute():
                        p_out = BID_SERVER_ROOT / p_out
                    local_dir = p_out / file_uid
                    if local_dir.exists():
                        files = list(local_dir.glob("*"))
                        if files:
                            logger.info(f"[清标服务] 直接读取本地文件: {files[0]}")
                            return files[0].read_bytes()
                    # 本地文件不存在，尝试网络请求
                    logger.warning(f"[清标服务] 本地文件不存在，回退到网络请求: {local_dir}")
        
        # 其他情况：直接使用 requests.get
        logger.info(f"[清标服务] 网络请求: {url[:80]}...")
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        return resp.content
        
    except Exception as e:
        logger.error(f"[清标服务] 文件获取失败: {url[:80]}... 错误: {e}")
        return None


async def _fetch_file_async(url: str, timeout: int = 300) -> Optional[bytes]:
    """
    异步获取文件内容，在线程池中执行同步操作，避免阻塞事件循环。
    """
    return await asyncio.to_thread(_fetch_file_directly_sync, url, timeout)


# 同步版本别名，保持向后兼容
_fetch_file_directly = _fetch_file_directly_sync


def calculate_file_hash(file_path: Path) -> str:
    """计算文件的 SHA256 哈希值"""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        # 分块读取,避免大文件占用过多内存
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def get_cached_md_path(pdf_path: Path) -> Optional[Path]:
    """
    根据 PDF 文件哈希值查找缓存的 MD 文件
    返回缓存的 MD 文件路径,如果不存在则返回 None
    """
    try:
        file_hash = calculate_file_hash(pdf_path)
        cached_md = MD_CACHE_DIR / f"{file_hash}.md"
        
        if cached_md.exists():
            logger.info(f"Found cached MD for {pdf_path.name} (hash: {file_hash[:8]}...)")
            return cached_md
        return None
    except Exception as e:
        logger.warning(f"Error checking MD cache: {e}")
        return None

def save_md_to_cache(pdf_path: Path, md_content: bytes) -> Optional[Path]:
    """
    将 MD 文件保存到全局缓存
    返回缓存文件路径
    """
    try:
        file_hash = calculate_file_hash(pdf_path)
        cached_md = MD_CACHE_DIR / f"{file_hash}.md"
        
        cached_md.write_bytes(md_content)
        logger.info(f"Saved MD to cache: {cached_md.name}")
        return cached_md
    except Exception as e:
        logger.error(f"Error saving MD to cache: {e}")
        return None

def extract_basic_info_from_md(md_path):
    """从Markdown文件提取前3页内容 (通过Page Split分割)"""
    try:
        with open(md_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # 假设分页符是 <--- Page Split --->
        pages = content.split('<--- Page Split --->')
        
        # 取前3页
        first_3_pages = pages[:3]
        
        return "\n<--- Page Split --->\n".join(first_3_pages)
    except Exception as e:
        logger.error(f"Error extracting first 3 pages from MD: {e}")
        return None

def convert_to_text(local_pdf_path, ocr_server_url, pdf_url=None, use_local_path=None):
    """
    Helper to convert PDF to Text via OCR (DeepSeek-OCR 新接口)
    使用全局缓存机制,避免重复转换相同的 PDF 文件
    
    新OCR接口特点（DeepSeek-OCR）：
    - POST /async/ocr/pdf/local 创建异步任务（本地路径模式）
    - POST /async/ocr/pdf 创建异步任务（文件上传模式）
    - GET /async/task/{task_id} 查询任务状态
    - GET /async/task/{task_id}/result 获取任务结果
    - 响应直接包含 full_content，无需额外下载
    
    Args:
        local_pdf_path: 本地PDF文件路径
        ocr_server_url: OCR服务地址 (默认 http://localhost:8810)
        pdf_url: PDF的URL地址（可选，文件上传模式时使用）
        use_local_path: 是否使用本地路径模式（OCR服务与本服务同机部署时设为True可避免网络传输）
                        如果不指定，则从环境变量 OCR_USE_LOCAL_PATH 读取
    """
    import time
    
    # Check for Mock OCR mode
    if _get_env("MOCK_OCR", "false").lower() in ("true", "1", "yes"):
        logger.info(f"[MOCK OCR] Skipping OCR for {local_pdf_path}, using mock data.")
        local_pdf_path = Path(local_pdf_path)
        txt_filename = local_pdf_path.stem + ".txt"
        md_filename = local_pdf_path.stem + ".md"
        local_txt_path = local_pdf_path.parent / txt_filename
        local_md_path = local_pdf_path.parent / md_filename
        
        mock_content = "# Mock OCR Content\n\nThis is a mock markdown content generated because MOCK_OCR is enabled.\n\n## Section 1\n\nSome sample text for extraction."
        # Try to read from a local mock file if it exists
        mock_file_path = _get_env("MOCK_OCR_FILE")
        if mock_file_path and Path(mock_file_path).exists():
             try:
                 mock_content = Path(mock_file_path).read_text(encoding="utf-8")
                 logger.info(f"[MOCK OCR] Loaded mock content from {mock_file_path}")
             except Exception as e:
                 logger.error(f"[MOCK OCR] Failed to read mock file: {e}")
        
        # output_dir logic placeholder
        output_dir = None
        if output_dir:
            out_path = Path(output_dir)
            out_path.mkdir(parents=True, exist_ok=True)
            md_path = out_path / f"{local_pdf_path.stem}.md"
            md_path.write_text(mock_content, encoding='utf-8')
            return str(md_path)
        
        # If no output_dir, save to same directory as local_pdf_path (consistent with non-mock behavior)
        if local_pdf_path:
             md_path = Path(local_pdf_path).parent / f"{Path(local_pdf_path).stem}.md"
             md_path.write_text(mock_content, encoding='utf-8')
             return str(md_path)

        return mock_content

    try:
        local_pdf_path = Path(local_pdf_path)
        # 输出为txt文件（但为了兼容，也保存为md）
        txt_filename = local_pdf_path.stem + ".txt"
        md_filename = local_pdf_path.stem + ".md"
        local_txt_path = local_pdf_path.parent / txt_filename
        local_md_path = local_pdf_path.parent / md_filename
        
        # 1. 检查本地是否已有文件
        if local_md_path.exists():
            logger.info(f"Local text file already exists: {local_md_path}")
            return str(local_md_path)
        if local_txt_path.exists():
            logger.info(f"Local text file already exists: {local_txt_path}")
            # 复制为md以保持兼容
            shutil.copy(local_txt_path, local_md_path)
            return str(local_md_path)
        
        # 2. 检查全局缓存
        cached_md = get_cached_md_path(local_pdf_path)
        if cached_md:
            try:
                # 确保目标目录存在
                local_md_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(cached_md, local_md_path)
                # 验证文件是否真正复制成功
                if local_md_path.exists():
                    logger.info(f"Copied cached text to local: {local_md_path}")
                    return str(local_md_path)
                else:
                    logger.warning(f"Cache copy appeared successful but file not found: {local_md_path}")
            except Exception as e:
                logger.warning(f"Failed to copy cached text: {e}, will proceed with OCR")

        # 3. 确定使用本地路径还是URL模式
        if use_local_path is None:
            use_local_path = _get_env("OCR_USE_LOCAL_PATH", "").lower() in ("true", "1", "yes")
        
        # 检查本地文件是否存在（本地路径模式必须）
        if not local_pdf_path.exists():
            logger.warning(f"本地文件不存在: {local_pdf_path}")
            use_local_path = False

        logger.info(f"Converting {local_pdf_path} to text via OCR (本地路径模式: {use_local_path})...")
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # 1. 创建OCR异步任务 - 新接口使用 /async/ocr/pdf/local
                if use_local_path and local_pdf_path.exists():
                    # 本地路径模式
                    payload = {
                        "file_path": str(local_pdf_path.resolve()),
                        "prompt_type": "document",
                        "skip_repeat": True
                    }
                    logger.info(f"OCR using local path: {local_pdf_path.resolve()}")
                    resp = requests.post(
                        f"{ocr_server_url}/async/ocr/pdf/local",
                        json=payload,
                        headers={"Content-Type": "application/json"},
                        timeout=600
                    )
                else:
                    # 文件上传模式
                    logger.info(f"OCR using file upload mode")
                    with open(local_pdf_path, 'rb') as f:
                        files = {'file': (local_pdf_path.name, f, 'application/pdf')}
                        data = {'prompt_type': 'document', 'skip_repeat': 'true'}
                        resp = requests.post(
                            f"{ocr_server_url}/async/ocr/pdf",
                            files=files,
                            data=data,
                            timeout=600
                        )
                
                resp.raise_for_status()
                resp_data = resp.json()
                
                if not resp_data.get('success'):
                    raise Exception(f"OCR task creation failed: {resp_data.get('message', 'Unknown error')}")
                
                task_id = resp_data.get('task_id')
                logger.info(f"OCR Task ID: {task_id}, Status: {resp_data.get('status')}")
                
                # 2. 轮询等待完成 - 新接口使用 /async/task/{task_id}
                start_wait = time.time()
                while True:
                    status_resp = requests.get(f"{ocr_server_url}/async/task/{task_id}", timeout=120)
                    status_data = status_resp.json()
                    status = status_data.get('status')
                    progress = status_data.get('progress', {})
                    
                    if status == 'completed':
                        logger.info(f"OCR completed! Task: {task_id}")
                        break
                    elif status == 'failed':
                        error_msg = status_data.get('error', '未知错误')
                        raise Exception(f"OCR failed: {error_msg}")
                    elif status == 'cancelled':
                        raise Exception("OCR task was cancelled")
                    
                    # 超时检查 (4小时)
                    if time.time() - start_wait > 14400:
                        raise Exception("OCR processing timed out (4h)")
                    
                    # 记录进度
                    elapsed = time.time() - start_wait
                    if int(elapsed) % 30 == 0:  # 每30秒记录一次
                        current = progress.get('current', 0) if isinstance(progress, dict) else 0
                        total = progress.get('total', 0) if isinstance(progress, dict) else 0
                        logger.info(f"OCR processing... [{elapsed:.0f}s] status={status} progress={current}/{total}")
                        
                    time.sleep(5)
                
                # 3. 获取任务结果 - 新接口使用 /async/task/{task_id}/result
                result_resp = requests.get(f"{ocr_server_url}/async/task/{task_id}/result", timeout=300)
                result_resp.raise_for_status()
                result_data = result_resp.json()
                
                if result_data.get('status') != 'completed':
                    raise Exception(f"OCR result not ready: {result_data.get('error', 'Unknown error')}")
                
                # 从结果中提取文本内容
                ocr_result = result_data.get('result', {})
                if isinstance(ocr_result, dict):
                    data_obj = ocr_result.get('data', {})
                    txt_content = data_obj.get('full_content', '')
                else:
                    txt_content = str(ocr_result)
                
                if not txt_content:
                    raise Exception("OCR returned empty content")
                
                txt_bytes = txt_content.encode('utf-8')
                
                # 保存到本地（txt和md都保存，保持兼容）
                with open(local_txt_path, 'wb') as f:
                    f.write(txt_bytes)
                with open(local_md_path, 'wb') as f:
                    f.write(txt_bytes)
                logger.info(f"Text saved to: {local_md_path}")
                
                # 保存到全局缓存
                save_md_to_cache(local_pdf_path, txt_bytes)
                
                return str(local_md_path)
                
            except Exception as e:
                logger.error(f"Error during OCR conversion (Attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(10)
                else:
                    return None
    except Exception as e:
        logger.error(f"Error in convert_to_text: {e}")
        return None


# 保持旧函数名的兼容性
def convert_to_md(local_pdf_path, ocr_server_url, pdf_url=None):
    """兼容旧代码的别名"""
    return convert_to_text(local_pdf_path, ocr_server_url, pdf_url)

async def process_bid_purification(template_path, model_url, model_key, model_id, pdf_urls):
    """
    1. 提取清标项
    2. 将PDF文件通过OCR转换为Markdown (如果尚未转换)
    3. 从Markdown中截取前3页内容，调用信息抽取服务提取基本信息 (information)
    4. 调用信息抽取服务提取字段类清标项 (带 information)
    """
    # OCR服务地址（DeepSeek-OCR，默认8810端口）
    ocr_server_url = _get_env("OCR_SERVER_URL", "http://localhost:8810")
    
    # Check if template_path is a URL and download if necessary
    temp_template_path = None
    original_template_path = template_path
    
    if template_path.startswith("http://") or template_path.startswith("https://"):
        # 尝试优先在本地查找模板文件
        local_template_found = False
        
        # 1. 先尝试从 route 参数中提取（处理 /fetch?route=/download/xxx/filename 格式）
        url_to_check = template_path
        try:
            parsed = urllib.parse.urlparse(template_path)
            if parsed.query:
                query_params = urllib.parse.parse_qs(parsed.query)
                route_param = query_params.get("route", [""])[0]
                if route_param:
                    url_to_check = urllib.parse.unquote(route_param)
                    logger.debug(f"Extracted route from template URL: {url_to_check}")
        except Exception as e:
            logger.debug(f"Failed to parse template URL query: {e}")
        
        # 2. 从 URL 或 route 中提取 file_hash
        match = re.search(r'/download/([a-fA-F0-9]{32})/', url_to_check)
        if not match:
            match = re.search(r'/download/([a-fA-F0-9]{32})', url_to_check)
        
        if match:
            tpl_hash = match.group(1)
            logger.info(f"Detected template hash from URL: {tpl_hash}")
            file_upload_out = _get_env("FILE_UPLOAD_OUT_DIR", str(BID_SERVER_ROOT / "out" / "file_upload"))
            p_out = Path(file_upload_out)
            if not p_out.is_absolute():
                p_out = BID_SERVER_ROOT / p_out
            local_tpl_dir = p_out / tpl_hash
            if local_tpl_dir.exists():
                cand_files = list(local_tpl_dir.glob("*"))
                if cand_files:
                    template_path = str(cand_files[0])
                    local_template_found = True
                    logger.info(f"✅ Found local template, skipping download: {template_path}")
                else:
                    logger.info(f"Local template dir exists but no files: {local_tpl_dir}")
            else:
                logger.info(f"Local template dir not found: {local_tpl_dir}")

        if not local_template_found:
             try:
                 logger.info(f"Downloading template from {template_path}...")
                 # Use bid_server temp dir
                 temp_dir = BID_SERVER_ROOT / "out" / "temp" / "bid_purification_templates"
                 temp_dir.mkdir(parents=True, exist_ok=True)
                 
                 # Extract filename from URL or use default
                 parsed_url = urllib.parse.urlparse(template_path)
                 filename = None
                 if parsed_url.query:
                     query_params = urllib.parse.parse_qs(parsed_url.query)
                     route_param = query_params.get("route", [""])[0]
                     if route_param:
                         filename = os.path.basename(urllib.parse.unquote(route_param))
                 if not filename or filename == "fetch":
                     filename = os.path.basename(parsed_url.path) or "template.xlsx"
                 filename = urllib.parse.unquote(filename)
                 
                 import uuid
                 temp_template_path = temp_dir / f"{uuid.uuid4().hex}_{filename}"
                 
                 # 使用异步获取函数，在线程池中执行避免阻塞事件循环
                 content = await _fetch_file_async(template_path, timeout=1200)
                 if content is None:
                     raise Exception("Failed to fetch template file")
                 
                 with open(temp_template_path, 'wb') as f:
                     f.write(content)
                 
                 template_path = str(temp_template_path)
                 logger.info(f"Template downloaded to {template_path}")
                 
             except Exception as e:
                 logger.error(f"Failed to download template from {original_template_path}: {e}")
                 return None

    try:
        # 1. 提取清标项（在线程池中执行，避免阻塞事件循环）
        logger.info("Step 1: Extracting bid purification items from template...")
        bid_items_json = await asyncio.to_thread(
            extract_bid_purification_items, template_path, model_url, model_key, model_id
        )
        if not bid_items_json or "error" in bid_items_json:
            logger.error(f"Error extracting bid items: {bid_items_json}")
            return None
    finally:
        # Cleanup temp template if created
        if temp_template_path and os.path.exists(temp_template_path):
            try:
                os.remove(temp_template_path)
            except Exception as e:
                logger.warning(f"Failed to remove temp template file {temp_template_path}: {e}")


    # 合并字段类清标项
    tech_field_items = list(set(bid_items_json.get("技术清标项-字段类", [])))
    biz_field_items = list(set(bid_items_json.get("商务清标项-字段类", [])))
    field_items = list(set(tech_field_items + biz_field_items))
    logger.info(f"Extracted field items: {field_items}")
    
    tech_summary_items = list(set(bid_items_json.get("技术清标项-总结类", [])))
    biz_summary_items = list(set(bid_items_json.get("商务清标项-总结类", [])))
    summary_items = list(set(tech_summary_items + biz_summary_items))
    logger.info(f"Extracted summary items: {summary_items}")

    # 准备本地文件映射
    # 这里我们重构逻辑：
    # 1. 如果是 /download/{uid}/... 格式，尝试去 file_upload_service 的目录找。
    # 2. 如果找不到，或者 URL 是其他的，尝试下载到 bid_server 内的临时目录。
    
    # 使用 bid_server 内部的临时目录，确保服务独立性
    temp_dir = BID_SERVER_ROOT / "out" / "temp" / "bid_purification_processing"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # pdf_files_map: key -> {"local_path": Path, "pdf_url": str}
    pdf_files_map = {}
    
    for pdf_url in pdf_urls:
        local_pdf_path = None
        file_hash = None
        
        # Try to detect if it is a local file upload URL
        # 1. 先尝试从 route 参数中提取（处理 /fetch?route=/download/xxx/filename 格式）
        url_to_check = pdf_url
        try:
            parsed = urllib.parse.urlparse(pdf_url)
            if parsed.query:
                query_params = urllib.parse.parse_qs(parsed.query)
                route_param = query_params.get("route", [""])[0]
                if route_param:
                    # 解码 route 参数
                    url_to_check = urllib.parse.unquote(route_param)
                    logger.debug(f"Extracted route from URL: {url_to_check}")
        except Exception as e:
            logger.debug(f"Failed to parse URL query: {e}")
        
        # 2. 从 URL 或 route 中提取 file_hash
        match = re.search(r'/download/([a-fA-F0-9]{32})/', url_to_check)
        if not match:
            # 尝试不带结尾斜杠的匹配（可能是 /download/{hash}/{filename} 格式）
            match = re.search(r'/download/([a-fA-F0-9]{32})', url_to_check)
        
        if match:
            file_hash = match.group(1)
            logger.info(f"Detected file hash from URL: {file_hash}")
            # Check local path from file_upload_service (assuming we are on same machine/volume)
            # Use environment variable for file_upload output directory
            file_upload_out = _get_env("FILE_UPLOAD_OUT_DIR", str(BID_SERVER_ROOT / "out" / "file_upload"))
            p_out = Path(file_upload_out)
            if not p_out.is_absolute():
                p_out = BID_SERVER_ROOT / p_out
            local_file_dir = p_out / file_hash
            if local_file_dir.exists():
                cand_files = []
                cand_files.extend(list(local_file_dir.glob("*.pdf")))
                cand_files.extend(list(local_file_dir.glob("*.docx")))
                cand_files.extend(list(local_file_dir.glob("*.doc")))
                if cand_files:
                    local_pdf_path = cand_files[0]
                    # 如果找到了本地文件，直接使用，不需要下载
                    logger.info(f"✅ Found local file, skipping download: {local_pdf_path}")
                else:
                    logger.info(f"Local dir exists but no matching files: {local_file_dir}")
            else:
                logger.info(f"Local dir not found: {local_file_dir}")
        
        # If not found locally, download it
        if not local_pdf_path:
            try:
                parsed_url = urllib.parse.urlparse(pdf_url)
                # 尝试从 route 参数中提取真实文件名（处理 /fetch?route=/download/xxx/filename.pdf 格式）
                filename = None
                if parsed_url.query:
                    query_params = urllib.parse.parse_qs(parsed_url.query)
                    route_param = query_params.get("route", [""])[0]
                    if route_param:
                        filename = os.path.basename(urllib.parse.unquote(route_param))
                if not filename or filename == "fetch":
                    filename = os.path.basename(parsed_url.path) or "downloaded_file.pdf"
                filename = urllib.parse.unquote(filename)
                # 确保文件名有扩展名
                if not Path(filename).suffix:
                    filename = filename + ".pdf"
                
                # Create a unique temp file
                import uuid
                temp_file = temp_dir / f"{uuid.uuid4().hex}_{filename}"
                
                logger.info(f"Downloading {pdf_url} to {temp_file}...")
                
                # 使用异步获取函数，在线程池中执行避免阻塞事件循环
                content = await _fetch_file_async(pdf_url, timeout=300)
                if content is None:
                    raise Exception("Failed to fetch PDF file")
                with open(temp_file, 'wb') as f:
                    f.write(content)
                local_pdf_path = temp_file
                file_hash = str(local_pdf_path) # Use path as key if no hash
                
            except Exception as e:
                logger.error(f"Failed to download {pdf_url}: {e}")
                continue
        
        if local_pdf_path:
            # 保存本地路径和原始URL（新OCR接口需要URL）
            pdf_files_map[file_hash or pdf_url] = {
                "local_path": local_pdf_path,
                "pdf_url": pdf_url
            }

    if not pdf_files_map:
        logger.error("No valid PDF files found locally or downloaded.")
        return None

    # Step 2: 转换所有PDF到MD，并提取基本信息 (Information)
    logger.info("Step 2: Converting PDF to MD and extracting basic information...")
    
    full_md_urls = []
    basic_info_files = [] # 存储临时生成的只包含前3页内容的MD文件路径
    basic_info_md_urls = []
    
    # We need a way to serve these local files back to the Extract API if Extract API is external.
    # However, Extract API seems to take URLs.
    # If Extract API is local (127.0.0.1:8806), we can use file:// ? No, usually http.
    # We need to construct URLs that Extract API can access.
    # Since we are all in `bid_server` now, we can use our own `/fetch` endpoint to serve these files!
    # But `/fetch` serves from `file_upload` or `template_recommend` logic.
    # We might need to extend `/fetch` or `file_upload` to serve temporary files or just assume
    # that `extract_api` can access `http://js1.blockelite.cn:26988/...` which routes back to us.
    
    # In the original code, it constructed URLs like:
    # http://js1.blockelite.cn:26988/fetch?port=7004&route=/download/{file_hash}/{encoded_filename}
    # This assumes the file is in file_upload service (port 7004).
    
    # If we downloaded the file to temp dir, `file_upload` service doesn't know about it.
    # So we should probably UPLOAD it to `file_upload` service first? 
    # Or just use the original URL if it was already remote.
    
    # BUT `convert_to_md` creates a new MD file locally. We need to serve this MD file.
    # The original code assumed the MD file is generated ALONGSIDE the PDF in `file_upload` directory,
    # so it could be served via the same `file_hash`.
    
    # If we are using `file_upload` directory structure (bid_server/out/file_upload/{uid}),
    # and `convert_to_md` saves there, then we can serve it.
    
    gateway_public_base = _get_env("GATEWAY_PUBLIC_BASE_URL", "http://js1.blockelite.cn:26988")

    for key, file_info in pdf_files_map.items():
        local_pdf_path = file_info["local_path"]
        original_pdf_url = file_info["pdf_url"]
        suffix = str(Path(local_pdf_path).suffix or "").lower()
        
        # If we downloaded to temp, we have a problem serving it via existing file_upload logic
        # unless we move it to file_upload out dir.
        # Let's try to verify if it is in file_upload dir.
        file_upload_out_env = _get_env("FILE_UPLOAD_OUT_DIR", str(BID_SERVER_ROOT / "out" / "file_upload"))
        # 确保使用绝对路径进行比较
        file_upload_out_check = Path(file_upload_out_env)
        if not file_upload_out_check.is_absolute():
            file_upload_out_check = BID_SERVER_ROOT / file_upload_out_check
        file_upload_out_check = str(file_upload_out_check.resolve())
        is_in_file_upload = file_upload_out_check in str(Path(local_pdf_path).resolve())
        
        file_uid = key if is_in_file_upload else None
        
        if not is_in_file_upload:
            # We need to expose this file. 
            # Ideally we upload it to file_upload service to get a UID.
            # But that might be slow.
            # For now, let's assume if it's not in file_upload, we skip or we fail?
            # Or we mock the URL if Extract API is local and supports file paths (unlikely).
            
            # Let's register it in file_upload service manually (by moving file).
            # This is a bit hacky but ensures we can serve it.
            import uuid
            new_uid = uuid.uuid4().hex
            file_upload_out = _get_env("FILE_UPLOAD_OUT_DIR", str(BID_SERVER_ROOT / "out" / "file_upload"))
            dest_dir = Path(f"{file_upload_out}/{new_uid}")
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(local_pdf_path, dest_dir / local_pdf_path.name)
            local_pdf_path = dest_dir / local_pdf_path.name
            file_uid = new_uid
            is_in_file_upload = True

        if suffix == ".pdf":
            # 新OCR接口需要传入pdf_url（在线程池中执行，避免阻塞事件循环）
            full_md_path = await asyncio.to_thread(
                convert_to_md, local_pdf_path, ocr_server_url, original_pdf_url
            )
            if full_md_path:
                md_filename = Path(full_md_path).name
                encoded_filename = urllib.parse.quote(md_filename)
                # Construct URL using our new gateway logic (no port)
                # route=/download/{uid}/{filename}
                md_url = f"{gateway_public_base}/fetch?route=/download/{file_uid}/{encoded_filename}"
                full_md_urls.append(md_url)
                
                first3_content = extract_basic_info_from_md(full_md_path)
                if first3_content:
                    temp_md_name = f"first3_{md_filename}"
                    temp_md_path = local_pdf_path.parent / temp_md_name
                    with open(temp_md_path, 'w', encoding='utf-8') as f:
                        f.write(first3_content)
                    basic_info_files.append(temp_md_path)
                    
                    encoded_temp_name = urllib.parse.quote(temp_md_name)
                    temp_md_url = f"{gateway_public_base}/fetch?route=/download/{file_uid}/{encoded_temp_name}"
                    basic_info_md_urls.append(temp_md_url)
                    
        elif suffix in {".doc", ".docx"}:
            filename = Path(local_pdf_path).name
            encoded_filename = urllib.parse.quote(filename)
            file_url = f"{gateway_public_base}/fetch?route=/download/{file_uid}/{encoded_filename}"
            full_md_urls.append(file_url)
            basic_info_md_urls.append(file_url)

    information_str = None
    if basic_info_md_urls:
        basic_items = ["项目名称", "招标方（甲方,必须为公司名）", "投标方（乙方,必须为公司名）"]
        payload = {
            "files": basic_info_md_urls,
            "items": basic_items,
            "api_key": model_key,
            "base_url": model_url,
            "model": model_id,
        }
        try:
            logger.info("Calling extract_service for basic info...")
            # 直接调用 extract_service 函数，避免 HTTP 开销
            response = await extract_service(payload)
            # 如果返回的是 JSONResponse，提取其内容
            if isinstance(response, JSONResponse):
                response = json.loads(response.body.decode("utf-8"))
            if isinstance(response, dict):
                data = response.get("data", {})
                info_parts = []
                for k, v in data.items():
                    val = v
                    if isinstance(v, list):
                        val = v[0] if v else ""
                    if val:
                        info_parts.append(f"{k}：{val}")
                if info_parts:
                    information_str = "\n".join(info_parts)
                    logger.info(f"Extracted Information:\n{information_str}")
        except Exception as e:
            logger.error(f"Error extracting basic info: {e}")
            
    # 清理临时文件 (Optional, maybe keep for debug)
    # for f in basic_info_files:
    #     try:
    #         if f.exists():
    #             f.unlink()
    #     except Exception:
    #         pass

    if not full_md_urls:
        return None

    # Step 3: 处理完整文件并提取字段类 (带 Information)
    logger.info("Step 3: Processing full documents with information context for field items...")
    
    field_payload = {
        "files": full_md_urls,
        "items": field_items,
        "information": information_str,
        "api_key": model_key,
        "base_url": model_url,
        "model": model_id,
    }
    
    field_result = None
    try:
        logger.info("Calling extract_service for field items...")
        # 直接调用 extract_service 函数，避免 HTTP 开销
        response = await extract_service(field_payload)
        # 如果返回的是 JSONResponse，提取其内容
        if isinstance(response, JSONResponse):
            response = json.loads(response.body.decode("utf-8"))
        field_result = response
        logger.info("extract_service call successful for field items.")
    except Exception as e:
        logger.error(f"Error calling extract_service for field items: {e}")
        return None
        
    summary_result = None
    if summary_items:
        logger.info("Step 4: Processing full documents for summary items...")
        summary_payload = {
            "files": full_md_urls,
            "items": summary_items,
            "information": information_str,
            "task": "summary",
            "api_key": model_key,
            "base_url": model_url,
            "model": model_id,
        }
        try:
            logger.info("Calling extract_service for summary items...")
            # 直接调用 extract_service 函数，避免 HTTP 开销
            response2 = await extract_service(summary_payload)
            # 如果返回的是 JSONResponse，提取其内容
            if isinstance(response2, JSONResponse):
                response2 = json.loads(response2.body.decode("utf-8"))
            summary_result = response2
            logger.info("extract_service call successful for summary items.")
        except Exception as e:
            logger.error(f"Error calling extract_service for summary items: {e}")
            summary_result = None

    field_data = {}
    field_blocks = {}
    if isinstance(field_result, dict):
        field_data = field_result.get("data") or {}
        field_blocks = field_result.get("blocks") or {}
    
    tech_field_data = {k: v for k, v in (field_data or {}).items() if k in set(tech_field_items)}
    biz_field_data = {k: v for k, v in (field_data or {}).items() if k in set(biz_field_items)}
    tech_field_blocks = {k: v for k, v in (field_blocks or {}).items() if k in set(tech_field_items)}
    biz_field_blocks = {k: v for k, v in (field_blocks or {}).items() if k in set(biz_field_items)}
    
    summary_data = {}
    if isinstance(summary_result, dict):
        summary_data = summary_result.get("data") or {}
    
    tech_summary_data = {k: v for k, v in (summary_data or {}).items() if k in set(tech_summary_items)}
    biz_summary_data = {k: v for k, v in (summary_data or {}).items() if k in set(biz_summary_items)}
    
    return {
        "技术清标项-字段类": {"data": tech_field_data, "blocks": tech_field_blocks},
        "商务清标项-字段类": {"data": biz_field_data, "blocks": biz_field_blocks},
        "技术清标项-总结类": tech_summary_data,
        "商务清标项-总结类": biz_summary_data,
    }

async def purify(payload: dict):
    template_path = str(payload.get("template_path", "")).strip()
    pdf_urls = payload.get("pdf_urls") or payload.get("files")
    if not template_path:
        return JSONResponse(status_code=400, content={"error": "missing_template_path"})
    if not isinstance(pdf_urls, list) or not pdf_urls:
        return JSONResponse(status_code=400, content={"error": "missing_pdf_urls"})
    urls: List[str] = [str(u).strip() for u in pdf_urls if str(u).strip()]
    if not urls:
        return JSONResponse(status_code=400, content={"error": "missing_pdf_urls"})

    model_url = str(payload.get("model_url") or _get_env("OPENAI_BASE_URL", None) or "https://dashscope.aliyuncs.com/compatible-mode/v1").strip()
    model_id = str(payload.get("model_id") or payload.get("model") or _get_env("OPENAI_MODEL", None) or "qwen3-32b").strip()
    model_key = str(
        payload.get("model_key")
        or payload.get("api_key")
        or _get_env("OPENAI_API_KEY", None)
        or _get_env("ZHIPU_API_KEY", None)
        or ""
    ).strip()

    try:
        result = await process_bid_purification(template_path, model_url, model_key, model_id, urls)
        if result is None:
            return JSONResponse(status_code=502, content={"error": "upstream_failed"})
        return result
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": "internal_error", "message": str(e)})
