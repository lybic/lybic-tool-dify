import asyncio
from collections.abc import Generator
from typing import Any

from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage
from dify_plugin.errors.tool import ToolProviderCredentialValidationError
from lybic import LybicClient, Sandbox, LybicAuth


async def get_screenshot(org_id: str, api_key: str, endpoint:str, sandbox_id: str) -> str:
    async with LybicClient(LybicAuth(org_id=org_id, api_key=api_key, endpoint=endpoint)) as client:
        sandbox_client = Sandbox(client)
        screenshot_url, _, _ = await sandbox_client.get_screenshot(sandbox_id=sandbox_id)
        return screenshot_url


class GetScreenshotTool(Tool):
    def _invoke(self, tool_parameters: dict[str, Any]) -> Generator[ToolInvokeMessage, None, None]:
        sandbox_id = tool_parameters.get("sandbox_id")
        if not sandbox_id:
            raise Exception("Error: sandbox_id is required.")

        org_id = self.runtime.credentials.get("lybic_organization_id")
        api_key = self.runtime.credentials.get("lybic_api_key")
        endpoint = self.runtime.credentials.get("lybic_api_endpoint", "https://api.lybic.cn")

        if not org_id or not api_key:
            raise ToolProviderCredentialValidationError("Error: Lybic credentials are not configured.")

        try:
            image_url = asyncio.run(get_screenshot(org_id, api_key, endpoint, sandbox_id))
            yield self.create_image_message(image_url)

        except Exception as e:
            raise Exception(f"Error getting screenshot: {e}")
