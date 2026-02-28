import os
import io
import json
import time
import asyncio
import contextlib
import logging
from pathlib import Path
from typing import List, Dict, Optional, Union
import requests
from fastapi.responses import JSONResponse
import traceback

# Setup logger
logger = logging.getLogger("bid_server.extract")

# bid_server 根目录（相对路径，可移植）
BID_SERVER_ROOT = Path(__file__).resolve().parent.parent

@contextlib.contextmanager
def _timed(step: str, **meta):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        if meta:
            try:
                m = json.dumps(meta, ensure_ascii=False, default=str)
            except Exception:
                m = str(meta)
            logger.info(f"time {step} {dt:.3f}s {m}")
        else:
            logger.info(f"time {step} {dt:.3f}s")

def _get_env(key: str, default: Optional[str] = None) -> Optional[str]:
    v = os.environ.get(key)
    return v if v else default

def _new_run_dir() -> Path:
    base = Path(__file__).parent / "logs"
    base.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = base / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir

def _write_run_log(text: str, run_dir: Path) -> str:
    fp = run_dir / "step2_combined.txt"
    fp.write_text(text or "", encoding="utf-8", errors="ignore")
    return str(fp)

def _write_text(run_dir: Path, name: str, text: str) -> None:
    p = run_dir / name
    p.write_text(text or "", encoding="utf-8", errors="ignore")

def _is_local_server_url(url: str) -> bool:
    """检查 URL 是否指向本服务器"""
    local_hosts = [
        "js1.blockelite.cn:26988",
        "127.0.0.1:8800",
        "localhost:8800",
    ]
    for host in local_hosts:
        if host in url:
            return True
    return False


def _fetch_local_file_content(url: str) -> Optional[bytes]:
    """
    直接读取本地文件内容，避免 HTTP 死锁。
    仅用于本服务器的 /fetch?route=... 类型的 URL
    支持路由格式: /file/{uid}/{filename}, /download/{uid}/{filename}
    """
    try:
        from urllib.parse import urlparse, parse_qs, unquote
        
        parsed = urlparse(url)
        qs = parse_qs(parsed.query or "")
        route_list = qs.get("route") or []
        
        if not route_list:
            return None
        
        rpath = unquote(route_list[0] or "")
        parts = [p for p in rpath.split("/") if p]
        
        # 查找路径前缀
        path_prefix = None
        if "file" in parts:
            path_prefix = "file"
        elif "download" in parts:
            path_prefix = "download"
        
        if path_prefix:
            idx = parts.index(path_prefix)
            if len(parts) >= idx + 3:
                uid = parts[idx + 1]
                filename = "/".join(parts[idx + 2:])
                
                # 在文件上传目录中查找
                file_upload_out = _get_env("FILE_UPLOAD_OUT_DIR", str(BID_SERVER_ROOT / "out" / "file_upload"))
                p_out = Path(file_upload_out)
                if not p_out.is_absolute():
                    p_out = BID_SERVER_ROOT / p_out
                
                local_path = p_out / uid / filename
                if local_path.exists():
                    logger.info(f"[抽取服务] 直接读取本地文件: {local_path}")
                    return local_path.read_bytes()
                
                # 尝试模糊匹配（只匹配 uid 目录下的任意文件）
                local_dir = p_out / uid
                if local_dir.exists():
                    files = list(local_dir.glob("*"))
                    if files:
                        logger.info(f"[抽取服务] 模糊匹配本地文件: {files[0]}")
                        return files[0].read_bytes()
                
                logger.warning(f"[抽取服务] 本地文件不存在: {local_path}")
        
        return None
    except Exception as e:
        logger.error(f"[抽取服务] 获取本地文件失败: {e}")
        return None


def _download_bytes(url: str, timeout: int = 300) -> Optional[bytes]:
    try:
        # 如果是本地服务器的 URL，直接读取本地文件，避免 HTTP 死锁
        if _is_local_server_url(url):
            local_content = _fetch_local_file_content(url)
            if local_content is not None:
                return local_content
            logger.warning(f"[抽取服务] 本地文件读取失败，回退到网络请求: {url[:80]}...")
        
        with _timed("http_get", timeout=timeout):
            r = requests.get(url, timeout=timeout)
        if r.status_code == 200:
            return r.content
        return None
    except Exception:
        return None

def _get_ext_from_url(url: str) -> str:
    try:
        from urllib.parse import urlparse, parse_qs, unquote
        u = urlparse(url)
        path = u.path or ""
        name = path.split("/")[-1].lower()
        cand = ""
        if "." in name:
            cand = name.split(".")[-1]
        qs = parse_qs(u.query or "")
        route = qs.get("route", [])
        if (not cand or name in {"fetch", "download"}) and route:
            route_path = unquote(route[0] or "")
            rname = route_path.split("/")[-1].lower()
            if "." in rname:
                cand = rname.split(".")[-1]
        return cand
    except Exception:
        return ""

def _openai_chat(api_key: str, base_url: str, model: str, messages: list, temperature: float = 0.0, max_tokens: int = 1200, run_dir: Optional[str] = None) -> Optional[str]:
    try:
        endpoint = base_url.rstrip("/")
        if not endpoint.endswith("/chat/completions"):
            endpoint = endpoint + "/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "n": 1,
            "stream": False,
            "enable_thinking": False,
        }
        if run_dir:
            try:
                _write_text(Path(run_dir), "prompt_llm_payload.json", json.dumps({"url": endpoint, "headers": {"Authorization": "Bearer ****", "Content-Type": "application/json", "Accept": "application/json"}, "body": payload}, ensure_ascii=False, indent=2))
            except Exception:
                pass
        r = requests.post(endpoint, headers=headers, json=payload, timeout=60)
        if r.status_code != 200:
            if run_dir:
                try:
                    _write_text(Path(run_dir), "prompt_llm_error.json", json.dumps({"status": r.status_code, "body": r.text}, ensure_ascii=False, indent=2))
                except Exception:
                    pass
            return None
        obj = r.json()
        ch = obj.get("choices") or []
        if not ch:
            return None
        msg = ch[0].get("message") or {}
        cnt = msg.get("content") or ""
        return cnt.strip()
    except Exception:
        return None

def _ensure_page_splits(text: Optional[str]) -> Optional[str]:
    t = text or ""
    if not t.strip():
        return text
    marker = "<--- Page Split --->"
    if marker in t:
        return t
    try:
        lim_str = _get_env("DOC_PAGE_SPLIT_CHAR_LIMIT", "2000") or "2000"
        limit = int(lim_str)
    except Exception:
        limit = 2000
    if limit <= 0:
        return t
    lines = t.splitlines(keepends=True)
    pages: List[str] = []
    buf: List[str] = []
    cur = 0
    def flush():
        nonlocal buf, cur
        if buf:
            pages.append("".join(buf))
            buf = []
            cur = 0
    for ln in lines:
        if len(ln) > limit:
            flush()
            s = ln
            while s:
                pages.append(s[:limit])
                s = s[limit:]
            continue
        if cur + len(ln) > limit and buf:
            flush()
        buf.append(ln)
        cur += len(ln)
    flush()
    if len(pages) <= 1:
        return t
    return f"\n{marker}\n".join([p.rstrip("\n") for p in pages])

def _to_md_from_docx_bytes(data: bytes) -> Optional[str]:
    try:
        import mammoth
        logger.info(f"docx_bytes_len {len(data) if isinstance(data, (bytes, bytearray)) else 0}")
        with _timed("docx_to_md_mammoth", bytes_len=len(data) if isinstance(data, (bytes, bytearray)) else 0):
            res = mammoth.convert_to_markdown(io.BytesIO(data))
        val = res.value
        try:
            logger.info(f"docx_text_preview {(val or '')[:200]}")
        except Exception:
            pass
        return _ensure_page_splits(val)
    except Exception as e:
        try:
            logger.info(f"docx_convert_error {str(e)}")
        except Exception:
            pass
        return None

def _to_md_from_doc_bytes(data: bytes) -> Optional[str]:
    tmp_dir = Path(__file__).parent / "logs"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / f"{int(time.time()*1000)}.doc"
    try:
        tmp_path.write_bytes(data)
        import pypandoc
        with _timed("doc_to_md_pypandoc", bytes_len=len(data) if isinstance(data, (bytes, bytearray)) else 0):
            md = pypandoc.convert_file(str(tmp_path), "md")
        return _ensure_page_splits(md)
    except Exception:
        return None
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass

def _ocr_pdf_to_text(pdf_url: str, server_url: Optional[str], output_dir: Optional[str] = None, filename: str = "output", local_path: Optional[str] = None) -> Optional[str]:
    """
    通过新OCR接口将PDF转换为文本（DeepSeek-OCR）
    新OCR接口：
    - POST /async/ocr/pdf/local 创建异步任务（本地路径模式）
    - POST /async/ocr/pdf 创建异步任务（文件上传模式）
    - GET /async/task/{task_id} 查询任务状态
    - GET /async/task/{task_id}/result 获取任务结果
    - 响应直接包含 full_content，无需额外下载
    
    Args:
        pdf_url: PDF的URL地址
        server_url: OCR服务地址
        output_dir: 输出目录
        filename: 输出文件名
        local_path: 本地PDF路径（如果OCR服务与本服务同机部署，可直接使用本地路径避免网络传输）
    """
    # Check for Mock OCR mode
    if _get_env("MOCK_OCR", "false").lower() in ("true", "1", "yes"):
        logger.info(f"[MOCK OCR] Skipping OCR for {pdf_url or local_path}, returning mock data.")
        mock_content = "# Mock OCR Content\n\nThis is a mock markdown content generated because MOCK_OCR is enabled.\n\n## Section 1\n\nSome sample text for extraction."
        
        # Try to read from a local mock file if it exists
        mock_file_path = _get_env("MOCK_OCR_FILE")
        if mock_file_path and Path(mock_file_path).exists():
             try:
                 mock_content = Path(mock_file_path).read_text(encoding="utf-8")
                 logger.info(f"[MOCK OCR] Loaded mock content from {mock_file_path}")
             except Exception as e:
                 logger.error(f"[MOCK OCR] Failed to read mock file: {e}")

        if output_dir:
            out_path = Path(output_dir)
            out_path.mkdir(parents=True, exist_ok=True)
            md_path = out_path / f"{filename}.md"
            md_path.write_text(mock_content, encoding='utf-8')
            return str(md_path)
        return mock_content

    # 判断是否使用本地路径模式
    use_local_path = _get_env("OCR_USE_LOCAL_PATH", "").lower() in ("true", "1", "yes")
    
    if use_local_path and local_path and Path(local_path).exists():
        logger.info(f"OCR converting (本地路径模式): {local_path}")
    else:
        logger.info(f"OCR converting (URL模式): {pdf_url}")
        use_local_path = False  # 回退到URL模式
    
    try:
        ocr_url = server_url or "http://localhost:8810"
        
        # Step 1: 创建OCR异步任务
        with _timed("ocr_create_task"):
            if use_local_path and local_path:
                # 本地路径模式 - 使用 /async/ocr/pdf/local
                payload = {
                    "file_path": str(Path(local_path).resolve()),
                    "prompt_type": "document",
                    "skip_repeat": True
                }
                resp = requests.post(
                    f"{ocr_url}/async/ocr/pdf/local",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=600
                )
            else:
                # URL模式 - 需要先下载PDF再上传
                logger.info(f"Downloading PDF from: {pdf_url}")
                
                # 如果是本地服务器的 URL，先尝试直接读取本地文件，避免 HTTP 死锁
                pdf_content = None
                if _is_local_server_url(pdf_url):
                    pdf_content = _fetch_local_file_content(pdf_url)
                    if pdf_content:
                        logger.info(f"[OCR] 直接从本地文件读取 PDF，大小: {len(pdf_content)} bytes")
                
                if pdf_content is None:
                    pdf_resp = requests.get(pdf_url, timeout=300)
                    pdf_resp.raise_for_status()
                    pdf_content = pdf_resp.content
                
                files = {'file': (f'{filename}.pdf', pdf_content, 'application/pdf')}
                data = {'prompt_type': 'document', 'skip_repeat': 'true'}
                resp = requests.post(
                    f"{ocr_url}/async/ocr/pdf",
                    files=files,
                    data=data,
                    timeout=600
                )
            
            resp.raise_for_status()
            resp_data = resp.json()
            
            if not resp_data.get('success'):
                logger.error(f"OCR task creation failed: {resp_data.get('message', 'Unknown error')}")
                return None
            
            task_id = resp_data.get('task_id')
            logger.info(f"OCR Task ID: {task_id}, Status: {resp_data.get('status')}")
        
        # Step 2: 轮询等待完成 - 新接口使用 /async/task/{task_id}
        start_wait = time.time()
        timeout = 14400  # 4小时
        with _timed("ocr_wait_completion"):
            while True:
                status_resp = requests.get(f"{ocr_url}/async/task/{task_id}", timeout=120)
                status_data = status_resp.json()
                status = status_data.get('status')
                progress = status_data.get('progress', {})
                
                if status == 'completed':
                    logger.info(f"OCR completed! Task: {task_id}")
                    break
                elif status == 'failed':
                    error_msg = status_data.get('error', '未知错误')
                    logger.error(f"OCR failed: {error_msg}")
                    return None
                elif status == 'cancelled':
                    logger.error("OCR task was cancelled")
                    return None
                
                if time.time() - start_wait > timeout:
                    logger.error("OCR timeout (4h)")
                    return None
                
                # 记录进度
                elapsed = time.time() - start_wait
                if int(elapsed) % 30 == 0:  # 每30秒记录一次
                    current = progress.get('current', 0) if isinstance(progress, dict) else 0
                    total = progress.get('total', 0) if isinstance(progress, dict) else 0
                    logger.info(f"OCR processing... [{elapsed:.0f}s] status={status} progress={current}/{total}")
                    
                time.sleep(5)
        
        # Step 3: 获取任务结果 - 新接口使用 /async/task/{task_id}/result
        with _timed("ocr_get_result"):
            result_resp = requests.get(f"{ocr_url}/async/task/{task_id}/result", timeout=300)
            result_resp.raise_for_status()
            result_data = result_resp.json()
            
            if result_data.get('status') != 'completed':
                logger.error(f"OCR result not ready: {result_data.get('error', 'Unknown error')}")
                return None
            
            # 从结果中提取文本内容
            ocr_result = result_data.get('result', {})
            if isinstance(ocr_result, dict):
                data_obj = ocr_result.get('data', {})
                text_content = data_obj.get('full_content', '')
            else:
                text_content = str(ocr_result)
            
            if not text_content:
                logger.error("OCR returned empty content")
                return None
            
            # 保存到文件
            if output_dir:
                out_path = Path(output_dir)
                out_path.mkdir(parents=True, exist_ok=True)
                md_path = out_path / f"{filename}.md"
                md_path.write_text(text_content, encoding='utf-8')
                return str(md_path)
            
            return text_content
            
    except Exception as e:
        logger.error(f"OCR error: {e}")
        return None


def _ocr_pdf_to_md_path(pdf_path: str, server_url: Optional[str], output_dir: Optional[str] = None) -> Optional[str]:
    """兼容旧代码的函数 - 使用新的DeepSeek-OCR接口"""
    logger.info(f"_ocr_pdf_to_md_path: Converting {pdf_path}")
    
    # Check for Mock OCR mode
    if _get_env("MOCK_OCR", "false").lower() in ("true", "1", "yes"):
        logger.info(f"[MOCK OCR] Skipping OCR for {pdf_path}, returning mock data path.")
        out_dir = output_dir or str(Path(__file__).parent / "logs" / time.strftime("%Y%m%d_%H%M%S"))
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        md_path = out_path / f"{Path(pdf_path).stem}.md"
        
        mock_content = "# Mock OCR Content\n\nThis is a mock markdown content generated because MOCK_OCR is enabled.\n\n## Section 1\n\nSome sample text for extraction."
         # Try to read from a local mock file if it exists
        mock_file_path = _get_env("MOCK_OCR_FILE")
        if mock_file_path and Path(mock_file_path).exists():
             try:
                 mock_content = Path(mock_file_path).read_text(encoding="utf-8")
                 logger.info(f"[MOCK OCR] Loaded mock content from {mock_file_path}")
             except Exception as e:
                 logger.error(f"[MOCK OCR] Failed to read mock file: {e}")
        
        md_path.write_text(mock_content, encoding='utf-8')
        return str(md_path)

    try:
        out_dir = output_dir or str(Path(__file__).parent / "logs" / time.strftime("%Y%m%d_%H%M%S"))
        ocr_url = server_url or "http://localhost:8810"
        
        # 使用本地路径模式调用新OCR接口
        payload = {
            "file_path": str(Path(pdf_path).resolve()),
            "prompt_type": "document",
            "skip_repeat": True
        }
        
        # 创建异步任务
        resp = requests.post(
            f"{ocr_url}/async/ocr/pdf/local",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=600
        )
        resp.raise_for_status()
        resp_data = resp.json()
        
        if not resp_data.get('success'):
            logger.error(f"OCR task creation failed: {resp_data.get('message')}")
            return None
        
        task_id = resp_data.get('task_id')
        logger.info(f"OCR Task ID: {task_id}")
        
        # 轮询等待完成
        start_wait = time.time()
        while True:
            status_resp = requests.get(f"{ocr_url}/async/task/{task_id}", timeout=120)
            status_data = status_resp.json()
            status = status_data.get('status')
            
            if status == 'completed':
                break
            elif status in ('failed', 'cancelled'):
                logger.error(f"OCR task {status}: {status_data.get('error')}")
                return None
            
            if time.time() - start_wait > 14400:  # 4小时超时
                logger.error("OCR timeout")
                return None
            
            time.sleep(5)
        
        # 获取结果
        result_resp = requests.get(f"{ocr_url}/async/task/{task_id}/result", timeout=300)
        result_resp.raise_for_status()
        result_data = result_resp.json()
        
        ocr_result = result_data.get('result', {})
        if isinstance(ocr_result, dict):
            text_content = ocr_result.get('data', {}).get('full_content', '')
        else:
            text_content = str(ocr_result)
        
        if text_content:
            out_path = Path(out_dir)
            out_path.mkdir(parents=True, exist_ok=True)
            md_path = out_path / f"{Path(pdf_path).stem}.md"
            md_path.write_text(text_content, encoding='utf-8')
            return str(md_path)
        
        return None
    except Exception as e:
        logger.error(f"OCR conversion failed: {e}")
        return None

def _pdf_url_to_md_text(url: str, server_url: Optional[str], output_dir: Optional[str], local_path: Optional[str] = None) -> Optional[str]:
    """
    将PDF URL转换为文本
    新OCR接口支持URL或本地路径模式
    
    Args:
        url: PDF的URL地址
        server_url: OCR服务地址
        output_dir: 输出目录
        local_path: 本地PDF路径（如果存在且启用本地模式，则避免网络传输）
    """
    filename = f"ocr_{int(time.time()*1000)}"
    with _timed("ocr_pdf_url_to_text"):
        result = _ocr_pdf_to_text(url, server_url, output_dir, filename, local_path=local_path)
    
    if result:
        # 如果返回的是文件路径
        if isinstance(result, str) and Path(result).exists():
            return Path(result).read_text(encoding="utf-8", errors="ignore")
        # 如果直接返回文本内容
        return result
    
    return None

def _extract_text_from_url(url: str, ocr_server_url: Optional[str], output_dir: Optional[str]) -> Optional[str]:
    def _map_gateway_download_to_local_path(u: str) -> Optional[str]:
        try:
            from urllib.parse import urlparse, parse_qs, unquote
            pu = urlparse(u)
            qs = parse_qs(pu.query or "")
            route_list = qs.get("route") or []
            if not route_list:
                return None
            rpath = unquote(route_list[0] or "")
            parts = [p for p in rpath.split("/") if p]
            
            # 支持两种 URL 格式:
            # - 旧格式: /download/{uid}/{filename}
            # - 新格式: /file/{uid}/{filename}
            path_prefix = None
            if "download" in parts:
                path_prefix = "download"
            elif "file" in parts:
                path_prefix = "file"
            
            if path_prefix:
                idx = parts.index(path_prefix)
                if len(parts) >= idx + 3:
                    hash_val = parts[idx + 1]
                    fname = parts[idx + 2]
                    # Use environment variable for file_upload output directory
                    file_upload_out = _get_env("FILE_UPLOAD_OUT_DIR", str(BID_SERVER_ROOT / "out" / "file_upload"))
                    p_out = Path(file_upload_out)
                    if not p_out.is_absolute():
                        p_out = BID_SERVER_ROOT / p_out
                    p = p_out / hash_val / fname
                    logger.info(f"[路径映射] 尝试本地路径: {p} (prefix={path_prefix})")
                    # 检查文件是否存在，如果不存在则尝试模糊匹配
                    if not p.exists():
                        logger.info(f"[路径映射] 文件不存在，尝试模糊匹配...")
                        # 尝试忽略 Hash，在 file_upload_out 目录下查找同名文件
                        try:
                            found = list(p_out.glob(f"*/{fname}"))
                            if found:
                                # 优先取第一个匹配的
                                p = found[0]
                                logger.info(f"[路径映射] 模糊匹配成功: {p}")
                        except Exception:
                            pass
                    return str(p)
            return None
        except Exception as e:
            logger.warning(f"[路径映射] 解析失败: {e}")
            return None
    with _timed("map_gateway_download_to_local_path"):
        lp = _map_gateway_download_to_local_path(str(url))
    if lp:
        try:
            ok = Path(lp).exists()
            logger.info(f"local_path {lp} {ok}")
            if ok:
                if lp.lower().endswith(".pdf"):
                    with _timed("local_pdf_to_md_text"):
                        # 传入本地路径，如果启用了 OCR_USE_LOCAL_PATH 则直接使用本地路径调用 OCR
                        t = _pdf_url_to_md_text(url, ocr_server_url, output_dir, local_path=lp)
                elif lp.lower().endswith(".docx"):
                    with _timed("local_docx_to_md_text"):
                        t = _to_md_from_docx_bytes(Path(lp).read_bytes())
                elif lp.lower().endswith(".doc"):
                    with _timed("local_doc_to_md_text"):
                        t = _to_md_from_doc_bytes(Path(lp).read_bytes())
                else:
                    try:
                        with _timed("local_read_text"):
                            t = Path(lp).read_text(encoding="utf-8", errors="ignore")
                    except Exception:
                        t = None
                #logger.info(f"local_text_preview {(t or '')[:200] if isinstance(t, str) else ''}")
                if t:
                    return t
        except Exception as e:
            try:
                logger.info(f"local_read_error {str(e)}")
            except Exception:
                pass
    with _timed("get_ext_from_url"):
        ext = _get_ext_from_url(url)
    if ext == "pdf":
        with _timed("remote_pdf_to_md_text"):
            t = _pdf_url_to_md_text(url, ocr_server_url, output_dir)
        logger.info(f"remote_pdf_text_preview {(t or '')[:200] if isinstance(t, str) else ''}")
        return t
    with _timed("download_file"):
        data = _download_bytes(url)
    if not data:
        logger.info("remote_download_failed")
        return None
    if ext == "docx":
        with _timed("remote_docx_to_md_text"):
            t = _to_md_from_docx_bytes(data)
        logger.info(f"remote_docx_text_preview {(t or '')[:200] if isinstance(t, str) else ''}")
        return t
    if ext == "doc":
        with _timed("remote_doc_to_md_text"):
            t = _to_md_from_doc_bytes(data)
        logger.info(f"remote_doc_text_preview {(t or '')[:200] if isinstance(t, str) else ''}")
        return t
    if ext in {"md", "markdown", "txt"}:
        try:
            with _timed("decode_text"):
                t = data.decode("utf-8", errors="ignore")
            logger.info(f"remote_text_preview {t[:200]}")
            return t
        except Exception as e:
            logger.info(f"remote_text_decode_error {str(e)}")
            return None
    logger.info(f"unsupported_ext {ext}")
    return None

def _combine_text(urls: List[str], texts: List[str]) -> str:
    blocks = []
    for u, t in zip(urls, texts):
        fn = u.split("?")[0].split("/")[-1]
        blocks.append(f"<{fn}>：\n{t or ''}")
    return "\n\n".join(blocks)

def _split_pages_from_combined(combined: str) -> List[Dict[str, str]]:
    import re
    pages: List[Dict[str, str]] = []
    it = list(re.finditer(r"<([^>]+)>：\n", combined))
    for i, m in enumerate(it):
        name = m.group(1)
        s = m.end()
        e = it[i + 1].start() if i + 1 < len(it) else len(combined)
        block = combined[s:e]
        parts = block.split("<--- Page Split --->")
        for idx, p in enumerate(parts):
            pages.append({"source": name, "page_index": str(idx + 1), "text": p})
    return pages

def _best_page_for_text(pages: List[Dict[str, str]], cand: str) -> Optional[Dict[str, str]]:
    best = None
    best_cnt = 0
    for pg in pages:
        cnt = (pg.get("text") or "").count(cand)
        if cnt > best_cnt:
            best_cnt = cnt
            best = pg
    if best:
        return best
    best_score=0
    for pg in pages:
        sc = _similar_ratio(cand, pg["text"])
        if sc > best_score:
            best_score = sc
            best = pg
    return best

def _select_one_candidate_with_llm(information: Optional[str], field_name: str, candidates: List[str], contexts: Dict[str, str], api_key: str, base_url: str, model: str, run_dir: Optional[str]) -> Optional[str]:
    info = information or ""
    lines = []
    lines.append(f"基础信息：\n{info}")
    lines.append(f"字段：{field_name}")
    lines.append("候选答案：")
    for c in candidates:
        lines.append(f"- {c}")
    lines.append("原文信息：")
    for c in candidates:
        ctx = contexts.get(c) or ""
        lines.append(f"候选 {c} 所在页：\n{ctx}")
    prompt = "\n".join(lines)
    messages = [
        {"role": "system", "content": "你是投标/招标文件抽取后的答案选择器。根据基础信息与原文页面内容，从候选答案中选择最准确的一项。仅输出候选答案之一，不做任何解释或其他字符。"},
        {"role": "user", "content": prompt},
    ]
    ans = _openai_chat(api_key, base_url, model, messages, temperature=0.0, max_tokens=64, run_dir=run_dir)
    if ans:
        logger.info(ans)
        cand_set = {c.strip() for c in candidates}
        a = ans.strip()
        if a in cand_set:
            return a
        best = None
        best_score = 0.0
        for c in candidates:
            sc = _similar_ratio(a, c)
            if sc > best_score:
                best_score = sc
                best = c
        return best
    return None

def _summary_with_llm(information: Optional[str], item_name: str, materials: str, api_key: str, base_url: str, model: str, run_dir: Optional[str]) -> str:
    info = information or ""
    prompt = "\n".join([
        f"基础信息：\n{info}",
        f"清标项：{item_name}",
        "材料：",
        materials or "",
        "请基于材料输出该清标项的总结结果（未必所有材料都包含清标项信息，需自行分析），要求：仅输出总结结果本身，不要输出任何解释、标题、序号或其他字符。字数100字以内",
    ])
    messages = [
        {"role": "system", "content": "你是投标/招标文件清标总结器。你只输出总结结果本身。"},
        {"role": "user", "content": prompt},
    ]
    ans = _openai_chat(api_key, base_url, model, messages, temperature=0.0, max_tokens=800, run_dir=run_dir)
    return (ans or "").strip()

def _refine_summary_items(combined: str, data: Dict[str, Union[str, List[str]]], information: Optional[str], api_key: str, base_url: str, model: str, run_dir: Optional[str]) -> Dict[str, str]:
    pages = _split_pages_from_combined(combined)
    page_map: Dict[str, Dict[int, str]] = {}
    for pg in pages:
        src = str(pg.get("source") or "")
        try:
            pi = int(str(pg.get("page_index") or "0").strip() or "0")
        except Exception:
            pi = 0
        if pi <= 0:
            continue
        page_map.setdefault(src, {})[pi] = (pg.get("text") or "")

    try:
        mw_str = _get_env("LLM_CONCURRENCY", "4") or "4"
        max_workers = int(mw_str)
    except Exception:
        max_workers = 4
    try:
        lim_str = _get_env("SUMMARY_CONTEXT_CHAR_LIMIT", "16000") or "16000"
        ctx_limit = int(lim_str)
    except Exception:
        ctx_limit = 16000
    try:
        per_str = _get_env("SUMMARY_MATERIAL_CHAR_LIMIT", "6000") or "6000"
        per_limit = int(per_str)
    except Exception:
        per_limit = 6000

    item_materials: Dict[str, str] = {}
    for k, v in (data or {}).items():
        item = str(k or "").strip()
        if not item:
            continue
        candidates: List[str] = []
        if isinstance(v, list):
            for x in v:
                s = str(x or "").strip()
                if s:
                    candidates.append(s)
        else:
            s = str(v or "").strip()
            if s:
                candidates.append(s)
        selected: Dict[str, set] = {}
        for c in candidates:
            pg = _best_page_for_text(pages, c)
            if not pg:
                continue
            src = str(pg.get("source") or "")
            try:
                pi = int(str(pg.get("page_index") or "0").strip() or "0")
            except Exception:
                pi = 0
            if src and pi > 0:
                selected.setdefault(src, set()).add(pi)
        if not selected:
            pg = _best_page_for_text(pages, item)
            if pg:
                src = str(pg.get("source") or "")
                try:
                    pi = int(str(pg.get("page_index") or "0").strip() or "0")
                except Exception:
                    pi = 0
                if src and pi > 0:
                    selected.setdefault(src, set()).add(pi)
        mats: List[str] = []
        mat_idx = 1
        for src in sorted(selected.keys()):
            idxs = sorted([i for i in selected[src] if isinstance(i, int) and i > 0])
            if not idxs:
                continue
            start = idxs[0]
            prev = idxs[0]
            def flush_range(a: int, b: int):
                nonlocal mat_idx
                texts = []
                for i in range(a, b + 1):
                    t = (page_map.get(src, {}).get(i) or "")
                    if t:
                        texts.append(t)
                if not texts:
                    return
                merged = "\n<--- Page Split --->\n".join(texts)
                merged = merged[:per_limit]
                if a == b:
                    header = f"材料{mat_idx}：\n{src} 第{a}页"
                else:
                    header = f"材料{mat_idx}：\n{src} 第{a}-{b}页"
                mats.append(f"{header}\n{merged}")
                mat_idx += 1
            for i in idxs[1:]:
                if i == prev + 1:
                    prev = i
                    continue
                flush_range(start, prev)
                start = i
                prev = i
            flush_range(start, prev)
        materials = "\n\n".join(mats)
        item_materials[item] = materials[:ctx_limit]

    out: Dict[str, str] = {}
    try:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {}
            for item, materials in item_materials.items():
                futures[item] = ex.submit(_summary_with_llm, information, item, materials, api_key, base_url, model, run_dir)
            for item, fut in futures.items():
                ans = ""
                try:
                    ans = (fut.result() or "").strip()
                except Exception as e:
                    logger.info(f"summary_with_llm_error {item} {repr(e)}")
                    traceback.print_exc()
                    ans = ""
                if run_dir:
                    try:
                        _write_text(Path(run_dir), f"summary_{item}.txt", ans)
                    except Exception:
                        pass
                out[item] = ans
        return out
    except Exception as e:
        logger.info(f"refine_summary_items_error {repr(e)}")
        traceback.print_exc()
        for item, materials in item_materials.items():
            out[item] = _summary_with_llm(information, item, materials, api_key, base_url, model, run_dir)
        return out

def _refine_multi_candidate_fields(combined: str, data: Dict[str, Union[str, List[str]]], information: Optional[str], api_key: str, base_url: str, model: str, run_dir: Optional[str]) -> Dict[str, Union[str, List[str]]]:
    pages = _split_pages_from_combined(combined)
    out: Dict[str, Union[str, List[str]]] = {}
    try:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        try:
            mw_str = _get_env("LLM_CONCURRENCY", "4") or "4"
            max_workers = int(mw_str)
        except Exception:
            max_workers = 4
        futures = []
        field_meta = {}
        for k, v in (data or {}).items():
            if isinstance(v, list) and len(v) >= 2:
                cands = [str(x).strip() for x in v if str(x).strip()]
                ctxs: Dict[str, str] = {}
                for c in cands:
                    pg = _best_page_for_text(pages, c)
                    if pg:
                        ctxs[c] = (pg.get("text") or "")[:4000]
                    else:
                        ctxs[c] = combined[:4000]
                field_meta[k] = {"cands": cands, "ctxs": ctxs}
            else:
                out[k] = v
        logger.info(field_meta)
        if field_meta:
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                for k, meta in field_meta.items():
                    futures.append((k, ex.submit(_select_one_candidate_with_llm, information, k, meta["cands"], meta["ctxs"], api_key, base_url, model, run_dir)))
                for k, fut in futures:
                    chosen = None
                    try:
                        chosen = fut.result()
                    except Exception as e:
                        logger.info(f"refine_multi_candidate_fields_error {repr(e)}")
                        traceback.print_exc()
                        chosen = None
                    cands = field_meta[k]["cands"]
                    if run_dir:
                        try:
                            _write_text(Path(run_dir), f"refine_{k}.json", json.dumps({"candidates": cands, "chosen": chosen}, ensure_ascii=False, indent=2))
                        except Exception:
                            pass
                    if chosen:
                        out[k] = chosen
                    else:
                        out[k] = cands[:1]
        return out
    except Exception as e:
        logger.info(f"refine_multi_candidate_fields_error {repr(e)}")
        traceback.print_exc()
        for k, v in (data or {}).items():
            if isinstance(v, list) and len(v) >= 2:
                cands = [str(x).strip() for x in v if str(x).strip()]
                ctxs: Dict[str, str] = {}
                for c in cands:
                    pg = _best_page_for_text(pages, c)
                    if pg:
                        ctxs[c] = (pg.get("text") or "")[:4000]
                    else:
                        ctxs[c] = combined[:4000]
                chosen = _select_one_candidate_with_llm(information, k, cands, ctxs, api_key, base_url, model, run_dir)
                if chosen:
                    out[k] = chosen
                else:
                    out[k] = cands[:1]
            else:
                out[k] = v
        return out

def _build_field_blocks(combined: str, data: Dict[str, Union[str, List[str]]], text_limit: int = 4000) -> Dict[str, Dict[str, str]]:
    pages = _split_pages_from_combined(combined)
    out: Dict[str, Dict[str, str]] = {}
    for k, v in (data or {}).items():
        val = v
        if isinstance(val, list):
            val = val[0] if val else ""
        val = str(val or "").strip()
        if not val:
            continue
        pg = _best_page_for_text(pages, val)
        if pg:
            out[k] = {
                "source": str(pg.get("source") or ""),
                "page_index": str(pg.get("page_index") or ""),
                "text": (pg.get("text") or "")[:text_limit],
            }
        else:
            out[k] = {"source": "", "page_index": "", "text": (combined or "")[:text_limit]}
    return out

def _build_prompt(items: List[str]) -> str:
    joined = "、".join(items)
    return (
        f"任务：从给定的中文合同文本中，按以下抽取项抽取对应的原文内容：{joined}。\n"
        f"要求：\n"
        f"- 仅抽取原文中属于每一抽取项的标题、段落或要点，不得改写；\n"
        f"- 按文档原始出现的顺序输出；\n"
        f"- 每一个抽取项使用对应的中文类别名作为 extraction_class，extraction_text 为该项的原文片段；\n"
        f"- 若某一项在原文中缺失，则不输出该项。"
    )

def build_system_prompt_from_items(items: List[str], api_key: str, base_url: str, model: str, run_dir: Optional[str] = None, information: Optional[str] = None) -> str:
    base = "你是投标/招标文件的结构化信息抽取提示词工程师。请根据给定抽取项，自动归纳每个字段的同义词、常见锚点与提取策略，并生成一段用于抽取的系统提示词。要求：仅抽取原文片段，不得改写；格式内容尽量精简，不带英文、md格式，可以换行；缺失项不输出；来源可为段落、标题、表格、附注、函件格式；当字段名未出现时，允许基于语义与角色映射推断但仍需引用原文中的实体。输出仅为提示词文本。"
    spec = f"抽取项：{('、'.join(items))}。请覆盖实体类、数值类、标识类、时间类、表格类字段的通用策略，并为角色类字段说明语义映射的通用规则。内容不应超过800字，若字段较多，可在保留字段的前提下适当简化。"
    messages = [{"role": "system", "content": base}, {"role": "user", "content": spec}]
    ans = _openai_chat(api_key, base_url, model, messages, temperature=0.0, max_tokens=800, run_dir=run_dir)
    if ans:
        if information:
            ans = f"【基本信息】\n{information}\n\n{ans}\n若材料没有抽取项答案，该项输出空字符串"
        return ans
    fallback = _build_prompt(items)
    if information:
        fallback = f"【基本信息】\n{information}\n\n{fallback}\n若材料没有抽取项答案，该项输出空字符串"
    return fallback

def _build_examples(items: List[str]):
    try:
        from langextract.core.data import ExampleData, Extraction
    except Exception:
        return None
    lines = []
    exts = []
    for it in items:
        lines.append(f"{it}：示例内容")
        exts.append(Extraction(extraction_class=it, extraction_text="示例内容"))
    return [ExampleData(text="\n".join(lines), extractions=exts)]

def build_examples_from_items(items: List[str], api_key: str, base_url: str, model: str, run_dir: Optional[str] = None):
    try:
        from langextract.core.data import ExampleData, Extraction
    except Exception:
        return None
    tmpl = (
        "基于投标/招标场景，生成一个示例文本与抽取结果。"
        "文本应包含与抽取项相关的典型要素（如角色实体、报价、项目名称、税率等，如适用）。"
        "抽取结果以JSON输出：{'text':'示例文本','extractions':[{'extraction_class':'字段','extraction_text':'原文片段'}...]};"
        f"字段列表：{items}。仅返回JSON。"
    )
    messages = [{"role": "system", "content": "你是结构化抽取的提示词与示例生成器。"}, {"role": "user", "content": tmpl}]
    content = _openai_chat(api_key, base_url, model, messages, temperature=0.2, max_tokens=1200, run_dir=run_dir)
    if not content:
        return None
    try:
        obj = json.loads(content)
        text = obj.get("text") or ""
        exts_json = obj.get("extractions") or []
        exts = []
        for e in exts_json:
            cls = str(e.get("extraction_class") or "").strip()
            val = str(e.get("extraction_text") or "").strip()
            if cls and val:
                exts.append(Extraction(extraction_class=cls, extraction_text=val))
        if not exts:
            return None
        return [ExampleData(text=text, extractions=exts)]
    except Exception:
        return None

def _examples_to_json(examples) -> str:
    try:
        arr = []
        for ex in examples or []:
            exts = []
            for e in getattr(ex, "extractions", []) or []:
                c = getattr(e, "extraction_class", "")
                t = getattr(e, "extraction_text", "")
                exts.append({"extraction_class": c, "extraction_text": t})
            arr.append({"text": getattr(ex, "text", ""), "extractions": exts})
        return json.dumps(arr, ensure_ascii=False, indent=2)
    except Exception:
        return ""

def _similar_ratio(a: str, b: str) -> float:
    try:
        import Levenshtein
        return Levenshtein.ratio(a, b)
    except Exception:
        from difflib import SequenceMatcher
        return SequenceMatcher(None, a, b).ratio()

def _filter_similar_to_examples(data: Dict[str, List[str]], examples, threshold: float, run_dir: Optional[str] = None) -> Dict[str, List[str]]:
    ex_texts: List[str] = []
    for ex in examples or []:
        for e in getattr(ex, "extractions", []) or []:
            t = getattr(e, "extraction_text", "")
            if t:
                ex_texts.append(str(t).strip())
    if not ex_texts:
        return data
    out: Dict[str, List[str]] = {}
    removed: Dict[str, List[str]] = {}
    for k, arr in (data or {}).items():
        kept: List[str] = []
        for s in arr or []:
            ss = (s or "").strip()
            if not ss:
                continue
            sim_high = False
            for et in ex_texts:
                if _similar_ratio(ss, et) >= threshold or _similar_ratio(ss.replace(" ", ""), et.replace(" ", "")) >= threshold:
                    sim_high = True
                    break
            if sim_high:
                removed.setdefault(k, []).append(ss)
            else:
                kept.append(ss)
        out[k] = kept
    if run_dir:
        try:
            _write_text(Path(run_dir), "filtered_by_examples.json", json.dumps(out, ensure_ascii=False, indent=2))
        except Exception:
            pass
    return out

def _dedup_by_levenshtein(arr: List[str], threshold: float) -> List[str]:
    if not arr:
        return []
    ratio = None
    try:
        import Levenshtein
        ratio = Levenshtein.ratio
    except Exception:
        from difflib import SequenceMatcher
        def ratio(a: str, b: str) -> float:
            return SequenceMatcher(None, a, b).ratio()
    out: List[str] = []
    for s in arr:
        ss = (s or "").strip()
        if not ss:
            continue
        dup = False
        for t in out:
            if ratio(ss, t) >= threshold or ratio(ss.replace(" ", ""), t.replace(" ", "")) >= threshold:
                dup = True
                break
        if not dup:
            out.append(ss)
    return out

def _postprocess_data(data: Dict[str, List[str]], threshold: float) -> Dict[str, Union[str, List[str]]]:
    res: Dict[str, Union[str, List[str]]] = {}
    for k, v in (data or {}).items():
        dv = _dedup_by_levenshtein(v or [], threshold)
        if len(dv) == 1:
            res[k] = dv[0]
        else:
            res[k] = dv
    return res

def _run_langextract(text: str, items: List[str], api_key: str, base_url: str, model: str, run_dir: Optional[str] = None, information: Optional[str] = None) -> Dict[str, List[str]]:
    try:
        import langextract as lx
        from langextract import factory
    except Exception as e:
        raise RuntimeError(f"LangExtract 未安装或导入失败: {e}")
    if len((text or "").strip()) < 32:
        return {k: [] for k in items}
    if api_key:
        os.environ.setdefault("OPENAI_API_KEY", api_key)
    if base_url:
        os.environ.setdefault("OPENAI_BASE_URL", base_url)
    with _timed("build_system_prompt_from_items", items=len(items or []), text_len=len(text or "")):
        prompt = build_system_prompt_from_items(items, api_key, base_url, model, run_dir, information)
    with _timed("build_examples_from_items", items=len(items or [])):
        examples = build_examples_from_items(items, api_key, base_url, model, run_dir) or _build_examples(items)
    if run_dir:
        d = Path(run_dir)
        _write_text(d, "prompt_system.txt", prompt or "")
        _write_text(d, "prompt_examples.json", _examples_to_json(examples))
    with _timed("build_model_config", provider="openai"):
        config = factory.ModelConfig(
            provider="openai",
            provider_kwargs={
                "api_key": api_key,
                "base_url": base_url,
                "model_id": model,
            },
        )
    try:
        max_char_buffer=len(prompt)*5
        logger.info(f"max_char_buffer {max_char_buffer}")
        with _timed("langextract.extract", max_char_buffer=max_char_buffer):
            result = lx.extract(
                text_or_documents=text,
                prompt_description=prompt,
                examples=examples,
                use_schema_constraints=False,
                config=config,
                max_char_buffer=max_char_buffer,
                show_progress=False,
            )
    except Exception as e:
        if run_dir:
            try:
                _write_text(Path(run_dir), "langextract_error.txt", str(e))
            except Exception:
                pass
        return {k: [] for k in items}
    data = {k: [] for k in items}
    if hasattr(result, "extractions") and isinstance(result.extractions, list):
        with _timed("collect_extractions", extraction_count=len(result.extractions)):
            for ext in result.extractions:
                cls = getattr(ext, "extraction_class", "")
                txt = getattr(ext, "extraction_text", "")
                if cls in data and txt:
                    data[cls].append(txt)
    try:
        th_str = _get_env("EXAMPLE_FILTER_THRESHOLD", "0.85") or "0.85"
        th = float(th_str)
    except Exception:
        th = 0.85
    with _timed("filter_similar_to_examples", threshold=th):
        data = _filter_similar_to_examples(data, examples, th, run_dir)
    logger.info(data)
    return data

async def extract(payload: dict):
    req_t0 = time.perf_counter()
    files = payload.get("files") or []
    items = payload.get("items") or []
    task = (payload.get("task") or "").strip()
    
    # 详细日志：记录请求参数
    logger.info(f"[抽取服务] ========== 开始处理 ==========")
    logger.info(f"[抽取服务] 文件数量: {len(files)}")
    logger.info(f"[抽取服务] 抽取字段: {items}")
    logger.info(f"[抽取服务] 任务类型: {task or '默认'}")
    for i, f in enumerate(files):
        logger.info(f"[抽取服务] 文件[{i}]: {f[:100]}{'...' if len(f) > 100 else ''}")
    
    if not isinstance(files, list) or not files:
        logger.info(f"[抽取服务] 错误: 缺少文件列表")
        logger.info(f"time request_total {(time.perf_counter()-req_t0):.3f}s {json.dumps({'error': 'missing_files'}, ensure_ascii=False)}")
        return JSONResponse(status_code=400, content={"error": "missing_files"})
    if not isinstance(items, list) or not items:
        logger.info(f"[抽取服务] 错误: 缺少抽取字段")
        logger.info(f"time request_total {(time.perf_counter()-req_t0):.3f}s {json.dumps({'error': 'missing_items'}, ensure_ascii=False)}")
        return JSONResponse(status_code=400, content={"error": "missing_items"})
    with _timed("new_run_dir"):
        run_dir = _new_run_dir()
    logger.info(f"[抽取服务] 运行目录: {run_dir}")
    
    # OCR服务地址（DeepSeek-OCR，默认8810端口）
    ocr_server_url = _get_env("OCR_BASE_URL") or 'http://localhost:8810'
    
    information = payload.get("information")
    api_key = payload.get("api_key") or _get_env("OPENAI_API_KEY") or _get_env("ZHIPU_API_KEY") or ""
    base_url = payload.get("base_url") or _get_env("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    model = payload.get("model") or _get_env("OPENAI_MODEL", "qwen3-32b")
    
    # 详细日志：记录模型配置
    logger.info(f"[抽取服务] 模型配置: model={model}, base_url={base_url[:50] if base_url else 'None'}...")
    logger.info(f"[抽取服务] API Key: {'已配置' if api_key else '未配置'} (长度: {len(api_key) if api_key else 0})")
    
    if not api_key:
        logger.info(f"[抽取服务] 错误: 缺少 API Key")
        logger.info(f"time request_total {(time.perf_counter()-req_t0):.3f}s {json.dumps({'error': 'missing_api_key'}, ensure_ascii=False)}")
        return JSONResponse(status_code=400, content={"error": "missing_api_key"})
    texts = []
    logger.info(f"[抽取服务] ========== 开始提取文件内容 ==========")
    for idx, u in enumerate(files):
        logger.info(f"[抽取服务] 正在提取文件[{idx}]: {u[:80]}...")
        with _timed("extract_text_from_url", index=idx):
            # 在线程池中执行，避免阻塞事件循环
            t = await asyncio.to_thread(_extract_text_from_url, u, ocr_server_url, str(run_dir))
        if t is None:
            texts.append("")
            logger.info(f"[抽取服务] 文件[{idx}] 提取结果: None (空)")
        else:
            texts.append(t)
            preview = t[:200].replace('\n', ' ') if t else ''
            logger.info(f"[抽取服务] 文件[{idx}] 提取成功: 长度={len(t)}, 预览=\"{preview}...\"")
        try:
            logger.info(f"file_text_len {idx} {len(texts[-1] or '')}")
        except Exception:
            pass
    
    logger.info(f"[抽取服务] ========== 合并文本 ==========")
    with _timed("combine_text", file_count=len(files or [])):
        combined = _combine_text(files, texts)
    
    combined_len = len(combined or "")
    combined_preview = (combined[:300].replace('\n', ' ') if combined else '')
    logger.info(f"[抽取服务] 合并后文本长度: {combined_len}")
    logger.info(f"[抽取服务] 合并后文本预览: \"{combined_preview}...\"")
    
    if combined_len == 0:
        logger.warning(f"[抽取服务] 警告: 合并后文本为空！无法进行抽取")
    
    with _timed("write_run_log", combined_len=len(combined or "")):
        _write_run_log(combined, run_dir)
    
    logger.info(f"[抽取服务] ========== 调用 LLM 抽取 ==========")
    logger.info(f"[抽取服务] 待抽取字段: {items}")
    try:
        with _timed("run_langextract"):
            # 在线程池中执行，避免阻塞事件循环
            data = await asyncio.to_thread(_run_langextract, combined, items, api_key, base_url, model, str(run_dir), information)
        logger.info(f"[抽取服务] LLM 抽取完成，原始结果: {json.dumps(data, ensure_ascii=False)[:500]}...")
    except Exception as e:
        logger.error(f"[抽取服务] LLM 抽取失败: {str(e)}")
        logger.error(f"[抽取服务] 错误详情: {traceback.format_exc()}")
        logger.info(f"time request_total {(time.perf_counter()-req_t0):.3f}s {json.dumps({'error': str(e)}, ensure_ascii=False)}")
        return JSONResponse(status_code=500, content={"error": str(e)})
    logger.info(f"[抽取服务] ========== 后处理数据 ==========")
    with _timed("postprocess_data", threshold=0.75):
        data = _postprocess_data(data, 0.75)
    logger.info(f"[抽取服务] 后处理完成: {json.dumps(data, ensure_ascii=False)[:500]}...")
    
    if task in {"summary", "summmy"}:
        try:
            with _timed("refine_summary_items"):
                # 在线程池中执行，避免阻塞事件循环
                data = await asyncio.to_thread(_refine_summary_items, combined, data, information, api_key, base_url, model, str(run_dir))
        except Exception as e:
            logger.warning(f"[抽取服务] refine_summary_items 失败: {e}")
        logger.info(f"[抽取服务] ========== 完成 (summary模式) ==========")
        logger.info(f"[抽取服务] 最终结果: {json.dumps(data, ensure_ascii=False)[:500]}...")
        logger.info(f"time request_total {(time.perf_counter()-req_t0):.3f}s {json.dumps({'file_count': len(files or []), 'item_count': len(items or []), 'task': task}, ensure_ascii=False)}")
        return {"data": data}
    try:
        with _timed("refine_multi_candidate_fields"):
            # 在线程池中执行，避免阻塞事件循环
            data = await asyncio.to_thread(_refine_multi_candidate_fields, combined, data, information, api_key, base_url, model, str(run_dir))
    except Exception as e:
        logger.warning(f"[抽取服务] refine_multi_candidate_fields 失败: {e}")
    
    if task == "contract":
        logger.info(f"[抽取服务] ========== 完成 (contract模式) ==========")
        # 统计抽取结果
        extracted_count = sum(1 for v in data.values() if v and (isinstance(v, list) and v or isinstance(v, str) and v.strip()))
        total_fields = len(items)
        logger.info(f"[抽取服务] 抽取统计: 成功={extracted_count}/{total_fields}")
        for k, v in data.items():
            if isinstance(v, list) and v:
                logger.info(f"[抽取服务]   • {k}: {v[0][:50] if v[0] else '(空)'}...")
            elif isinstance(v, str) and v.strip():
                logger.info(f"[抽取服务]   • {k}: {v[:50]}...")
            else:
                logger.info(f"[抽取服务]   • {k}: (未抽取到)")
        logger.info(f"time request_total {(time.perf_counter()-req_t0):.3f}s {json.dumps({'file_count': len(files or []), 'item_count': len(items or []), 'task': task}, ensure_ascii=False)}")
        return {"data": data}
    with _timed("build_field_blocks"):
        blocks = _build_field_blocks(combined, data)
    logger.info(f"[抽取服务] ========== 完成 (默认模式) ==========")
    logger.info(f"time request_total {(time.perf_counter()-req_t0):.3f}s {json.dumps({'file_count': len(files or []), 'item_count': len(items or []), 'task': task}, ensure_ascii=False)}")
    return {"data": data, "blocks": blocks}
