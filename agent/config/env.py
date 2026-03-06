import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def _first_env(*names: str):
    for n in names:
        v = os.getenv(n)
        if v is not None and str(v).strip() != "":
            return v
    return None

def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return str(value).strip().lower() in ("1", "true", "yes", "y", "on")

# Reasoning LLM configuration (for complex reasoning tasks)
REASONING_MODEL = os.getenv("REASONING_MODEL", "o1-mini")
REASONING_BASE_URL = _first_env(
    "REASONING_BASE_URL",
    "KIMI_BASE_URL",
    "KIMI_API_BASE",
    "MOONSHOT_BASE_URL",
    "MOONSHOT_API_BASE",
)
REASONING_API_KEY = _first_env(
    "REASONING_API_KEY",
    "KIMI_API_KEY",
    "MOONSHOT_API_KEY",
)

# Non-reasoning LLM configuration (for straightforward tasks)
BASIC_MODEL = os.getenv("BASIC_MODEL", "gpt-4o")
BASIC_BASE_URL = _first_env(
    "BASIC_BASE_URL",
    "KIMI_BASE_URL",
    "KIMI_API_BASE",
    "MOONSHOT_BASE_URL",
    "MOONSHOT_API_BASE",
)
BASIC_API_KEY = _first_env(
    "BASIC_API_KEY",
    "KIMI_API_KEY",
    "MOONSHOT_API_KEY",
)

# Vision-language LLM configuration (for tasks requiring visual understanding)
VL_MODEL = os.getenv("VL_MODEL", "gpt-4o")
VL_BASE_URL = os.getenv("VL_BASE_URL")
VL_API_KEY = os.getenv("VL_API_KEY")

# Coding LLM configuration (for coding tasks)
CODING_MODEL = os.getenv("CODING_MODEL", "kimi-k2.5")
CODING_BASE_URL = os.getenv("CODING_BASE_URL")
CODING_API_KEY = os.getenv("CODING_API_KEY")

# Chrome Instance configuration
CHROME_INSTANCE_PATH = os.getenv("CHROME_INSTANCE_PATH")

# LLM Streaming configuration
# Set to "false" to disable streaming for DeepSeek (useful for unstable networks)
DEEPSEEK_STREAMING = os.getenv("DEEPSEEK_STREAMING", "true").lower() == "true"

# NCBI Entrez configuration （开发测试时可以先写死）
NCBI_EMAIL = os.getenv("NCBI_EMAIL")  # required, sample：maxliu2022sz@gmail.com
NCBI_API_KEY = os.getenv("NCBI_API_KEY")  # optional, sample：sk-1234567890
NCBI_DEFAULT_DB = os.getenv("NCBI_DEFAULT_DB")  # optional, sample：pubmed
NCBI_MAX_RESULTS = os.getenv("NCBI_MAX_RESULTS")  # optional, sample：10

# Skills configuration
SKILLS_ENABLED = _env_bool("SKILLS_ENABLED", False)
SKILLS_AUTO_RELOAD = _env_bool("SKILLS_AUTO_RELOAD", True)
SKILLS_EXTRA_DIRS = os.getenv("SKILLS_EXTRA_DIRS")
BIOAGENT_WORKSPACE = os.getenv("BIOAGENT_WORKSPACE")
SKILLS_PROMPT_TARGETS = os.getenv("SKILLS_PROMPT_TARGETS", "researcher")
EXPERT_TOOLS_MODE = os.getenv("EXPERT_TOOLS_MODE", "full")
CODER_TOOLS_MODE = os.getenv("CODER_TOOLS_MODE", EXPERT_TOOLS_MODE)
