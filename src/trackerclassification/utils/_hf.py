from huggingface_hub import HfApi

import logging
LOGGER = logging.getLogger(__name__)

from ..config.secrets import HF_TOKEN


def repo_exists(repo_id: str) -> bool:
    api = HfApi(token=HF_TOKEN)
    try:
        api.repo_info(repo_id)
        return True
    except Exception:
        LOGGER.error("HF Repository %s does not exist or could not be accessed.", repo_id)
        raise