"""API/SSE 服务层（M6）。

对外暴露：
  - create_app(): FastAPI 应用工厂
  - AsyncEngine: 异步推理引擎（engine 模块提供，此处 re-export）
"""

from inferlite.engine.async_engine import AsyncEngine
from inferlite.server.app import create_app

__all__ = ["AsyncEngine", "create_app"]
