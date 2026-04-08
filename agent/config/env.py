from __future__ import annotations

"""MedAI 的环境变量引导模块。

这个文件虽然很小，但它承担了整个运行时配置初始化的入口职责：

1. 在模块导入阶段尽早加载 ``MedAI/.env``，避免每个脚本都重复写一遍
   环境变量加载逻辑。
2. 兼容两种运行环境：
   - 装了 ``python-dotenv`` 时，优先使用库来解析 ``.env``；
   - 没装该依赖时，退回到项目内置的轻量解析逻辑。
3. 对外暴露一组“已经按优先级归一化”的配置常量，供其他模块直接导入。

内置 fallback 解析器是有意保持克制的，只覆盖本项目当前实际需要的
``.env`` 语法子集：``KEY=VALUE``、可选引号、空行和 ``#`` 注释。
"""

import os
from pathlib import Path

try:
    # ``python-dotenv`` 在本项目里是可选依赖。
    # 如果环境里已经安装，就优先交给它处理，因为它对边界情况的支持更完整。
    from dotenv import load_dotenv
except Exception:
    # 某些轻量环境只装了核心依赖，没有安装 ``python-dotenv``。
    # 这里不要让导入失败，而是交给下面的 fallback 解析逻辑继续工作。
    load_dotenv = None


def load_dotenv_if_present(env_path: str | Path | None = None, *, overwrite: bool = False) -> None:
    """如果目标 ``.env`` 存在，就把其中的键值对加载进当前进程环境。

    这个函数负责统一处理“从哪里找 ``.env``”以及“是否允许覆盖已有环境变量”。
    默认行为是：

    - 未传 ``env_path`` 时，自动读取项目根目录 ``MedAI/.env``；
    - 已存在于 ``os.environ`` 的变量优先级更高，不会被文件覆盖；
    - 如果环境里装了 ``python-dotenv``，优先使用它；
    - 否则退回到本文件里的简化解析器。

    参数
    ----
    env_path:
        可选的 ``.env`` 路径。为空时，默认定位到项目根目录下的 ``.env``。
    overwrite:
        是否允许 ``.env`` 中的值覆盖当前进程里已经存在的环境变量。
        默认为 ``False``，也就是“shell 显式传入的值优先”。

    示例
    ----
    1. 使用项目默认的 ``MedAI/.env``：

        >>> load_dotenv_if_present()

    2. 指定自定义配置文件，并允许覆盖：

        >>> load_dotenv_if_present("D:/tmp/custom.env", overwrite=True)

    3. 在其他模块中先加载，再读取变量：

        >>> load_dotenv_if_present()
        >>> os.getenv("OPENAI_API_KEY")
    """

    if env_path is None:
        # ``env.py`` 位于 ``MedAI/agent/config/``，
        # 因此 ``parents[2]`` 正好回到项目根目录 ``MedAI/``。
        env_path = Path(__file__).resolve().parents[2] / ".env"
    else:
        env_path = Path(env_path)
        if env_path.exists() and env_path.is_dir():
            env_path = env_path / ".env"

    # ``.env`` 缺失不应该报错。
    # 有些部署环境本来就是通过系统环境变量注入配置，而不是依赖磁盘文件。
    if not env_path.exists():
        return

    if load_dotenv is not None:
        # 如果第三方库可用，就优先走库逻辑。
        load_dotenv(dotenv_path=env_path, override=overwrite)
        return

    # fallback 解析器：仅覆盖本项目实际会用到的最小语法集合。
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if not overwrite and key in os.environ:
            continue

        # 兼容以下两种写法，并统一去掉包裹引号：
        #   OPENAI_BASE_URL=https://...
        #   OPENAI_BASE_URL="https://..."
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]

        os.environ[key] = value


# 模块导入时立即尝试加载 ``.env``。
# 这样其他配置常量在定义时，就能直接从已经准备好的环境中取值。
load_dotenv_if_present()


def _first_env(*names: str, default: str | None = None) -> str | None:
    """按优先级顺序，返回第一个“存在且非空”的环境变量值。

    这个函数的作用是把“同一类配置可能有多个别名”的优先级规则收口到一处，
    避免不同模块各自实现一套查找逻辑。

    参数
    ----
    *names:
        按优先级从高到低传入的环境变量名列表。
    default:
        当所有变量都不存在或为空字符串时使用的默认值。

    返回
    ----
    str | None
        第一个命中的非空值；若都未命中，则返回 ``default``。

    示例
    ----
    1. 读取多个候选变量中的第一个：

        >>> _first_env("BASIC_API_KEY", "OPENAI_API_KEY")

    2. 给一个明确的兜底值：

        >>> _first_env("PLANNER_MODEL", "REASONING_MODEL", default="rule-based-planner")

    3. 如果前一个变量为空字符串，会继续向后找：

        >>> _first_env("EMPTY_VAR", "OPENAI_MODEL", default="gpt-4o-mini")
    """

    for name in names:
        value = os.getenv(name)
        if value is not None and str(value).strip():
            return value
    return default


# 模型通道路由说明：
# - ``PLANNER_MODEL``：偏规划/推理类角色使用；
# - ``TESTER_MODEL``：偏测试/验证类角色使用；
# - ``BASIC_MODEL``、``CODING_MODEL``：主运行通道的基础默认模型。
# 这些默认值主要是为了让本地测试或未配置外部模型的环境仍然可以启动。
PLANNER_MODEL = _first_env("PLANNER_MODEL", "REASONING_MODEL", default="rule-based-planner")
TESTER_MODEL = _first_env("TESTER_MODEL", "BASIC_MODEL", default="rule-based-tester")
BASIC_MODEL = _first_env("BASIC_MODEL", default="rule-based-basic")
CODING_MODEL = _first_env("CODING_MODEL", default="rule-based-calculator")

# 不同通道可选的 Base URL。
# 如果后续需要把不同类型请求打到不同的 OpenAI 兼容网关，这里就是统一入口。
BASIC_BASE_URL = _first_env("BASIC_BASE_URL")
CODING_BASE_URL = _first_env("CODING_BASE_URL")

# API Key 的优先级与 Base URL 类似：
# 先找通道专属 Key，再回退到通用 ``OPENAI_API_KEY``。
# 这样既支持细粒度配置，也兼容“全项目共用一个 Key”的简单场景。
BASIC_API_KEY = _first_env("BASIC_API_KEY", "OPENAI_API_KEY")
CODING_API_KEY = _first_env("CODING_API_KEY", "OPENAI_API_KEY")

# 某些 DeepSeek 兼容接口会把 streaming 开关以字符串方式传入。
# 这里统一做一次标准化，后面业务代码就能直接按布尔值使用。
DEEPSEEK_STREAMING = str(os.getenv("DEEPSEEK_STREAMING", "false")).strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
    "on",
}
