from .nodes import Sapiens2Extension
async def comfy_entrypoint() -> Sapiens2Extension:
    return Sapiens2Extension()
