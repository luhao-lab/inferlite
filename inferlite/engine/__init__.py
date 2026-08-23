"""Engine-facing protocols and runtime helpers.

这个包对外暴露 engine 层的公共 API。外部代码优先使用：

    from inferlite.engine import EngineCore, LLMModel, generate

而不是依赖内部文件路径：

    from inferlite.engine.engine import EngineCore, generate
    from inferlite.engine.context import LLMModel
"""

from inferlite.engine.async_engine import AsyncEngine
from inferlite.engine.context import LLMModel
from inferlite.engine.engine import EngineCore, generate
from inferlite.engine.metrics import MetricsCollector

__all__ = ["EngineCore", "LLMModel", "generate", "MetricsCollector", "AsyncEngine"]
