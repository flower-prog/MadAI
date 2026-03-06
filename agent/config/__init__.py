from .env import (
    # Reasoning LLM
    REASONING_MODEL,
    REASONING_BASE_URL,
    REASONING_API_KEY,
    # Basic LLM
    BASIC_MODEL,
    BASIC_BASE_URL,
    BASIC_API_KEY,
    # Vision-language LLM
    VL_MODEL,
    VL_BASE_URL,
    VL_API_KEY,
    # Coding LLM
    CODING_MODEL,
    CODING_BASE_URL,
    CODING_API_KEY,
    # Other configurations
    CHROME_INSTANCE_PATH,
    DEEPSEEK_STREAMING,
    NCBI_EMAIL,
    NCBI_API_KEY,
    NCBI_DEFAULT_DB,
    NCBI_MAX_RESULTS,
    # Skills configuration
    SKILLS_ENABLED,
    SKILLS_AUTO_RELOAD,
    SKILLS_EXTRA_DIRS,
    BIOAGENT_WORKSPACE,
    SKILLS_PROMPT_TARGETS,
    EXPERT_TOOLS_MODE,
    CODER_TOOLS_MODE,
)
from .tools import TAVILY_MAX_RESULTS

# Team configuration
TEAM_MEMBERS = ["expert", "researcher", "coder", "browser", "reporter"]

__all__ = [
    # Reasoning LLM
    "REASONING_MODEL",
    "REASONING_BASE_URL",
    "REASONING_API_KEY",
    # Basic LLM
    "BASIC_MODEL",
    "BASIC_BASE_URL",
    "BASIC_API_KEY",
    # Vision-language LLM
    "VL_MODEL",
    "VL_BASE_URL",
    "VL_API_KEY",
    # Coding LLM
    "CODING_MODEL",
    "CODING_BASE_URL",
    "CODING_API_KEY",
    # Other configurations
    "TEAM_MEMBERS",
    "TAVILY_MAX_RESULTS",
    "CHROME_INSTANCE_PATH",
    "DEEPSEEK_STREAMING",
    "NCBI_EMAIL",
    "NCBI_API_KEY",
    "NCBI_DEFAULT_DB",
    "NCBI_MAX_RESULTS",
    # Skills configuration
    "SKILLS_ENABLED",
    "SKILLS_AUTO_RELOAD",
    "SKILLS_EXTRA_DIRS",
    "BIOAGENT_WORKSPACE",
    "SKILLS_PROMPT_TARGETS",
    "EXPERT_TOOLS_MODE",
    "CODER_TOOLS_MODE",
]
