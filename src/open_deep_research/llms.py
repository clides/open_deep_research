from typing import Any, Dict, List, Optional

from langchain_core.callbacks import (AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import Runnable
from openai import OpenAI, AsyncOpenAI

class ChatOpenRouter(BaseChatModel):
    """A custom chat model for OpenRouter."""

    model_name: str
    api_key: str
    site_url: Optional[str] = None
    site_name: Optional[str] = None

    @property
    def _llm_type(self) -> str:
        return "openrouter"

    

    def _generate(
        self, messages: List[BaseMessage], stop: Optional[List[str]] = None, run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any
    ) -> ChatResult:
        client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
        )

        extra_headers = {}
        if self.site_url:
            extra_headers["HTTP-Referer"] = self.site_url
        if self.site_name:
            extra_headers["X-Title"] = self.site_name

        response = client.chat.completions.create(
            extra_headers=extra_headers,
            model=self.model_name,
            messages=[{"role": m.type, "content": m.content} for m in messages],
            **kwargs,
        )

        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=response.choices[0].message.content))])

    async def _agenerate(
        self, messages: List[BaseMessage], stop: Optional[List[str]] = None, run_manager: Optional[AsyncCallbackManagerForLLMRun] = None, **kwargs: Any
    ) -> ChatResult:
        client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
        )

        extra_headers = {}
        if self.site_url:
            extra_headers["HTTP-Referer"] = self.site_url
        if self.site_name:
            extra_headers["X-Title"] = self.site_name

        response = await client.chat.completions.create(
            extra_headers=extra_headers,
            model=self.model_name,
            messages=[{"role": m.type, "content": m.content} for m in messages],
            **kwargs,
        )

        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=response.choices[0].message.content))])
