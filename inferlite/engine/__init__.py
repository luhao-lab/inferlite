"""Engine-facing protocols and runtime helpers.

这个包对外暴露 engine 层的公共 API。外部代码优先使用：

    from inferlite.engine import EngineCore, LLMModel, generate

而不是依赖内部文件路径：

    from inferlite.engine.core import EngineCore, generate
    from inferlite.engine.protocol import LLMModel
"""

from inferlite.engine.core import EngineCore, generate
from inferlite.engine.metrics import MetricsCollector
from inferlite.engine.protocol import LLMModel

__all__ = ["EngineCore", "LLMModel", "generate", "MetricsCollector"]
