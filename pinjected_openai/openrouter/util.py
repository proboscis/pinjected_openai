from typing import Optional, List, Dict, Any, Callable, Awaitable, Protocol

import PIL
import httpx
import json_repair
from injected_utils.injected_cache_utils import sqlite_dict, async_cached
from openai import AsyncOpenAI
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletion
from pinjected import instance, design, IProxy, injected
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt

from pinjected_openai.compatibles import a_openai_compatible_llm
from pinjected_openai.vision_llm import to_content


# from vision_llm import a_vision_llm__gpt4o


class OpenRouterArchitecture(BaseModel):
    modality: str
    tokenizer: str
    instruct_type: Optional[str] = None


class OpenRouterModelPricing(BaseModel):
    prompt: str
    completion: str
    image: str
    request: str

    def calc_cost(self, usage: CompletionUsage):
        completion_cost = usage.completion_tokens * float(self.completion)
        prompt_cost = usage.prompt_tokens * float(self.prompt)
        return dict(
            completion=completion_cost,
            prompt=prompt_cost,
        )

    def calc_cost_dict(self, usage: dict):
        completion_cost = usage['completion_tokens'] * float(self.completion)
        prompt_cost = usage['prompt_tokens'] * float(self.prompt)
        return dict(
            completion=completion_cost,
            prompt=prompt_cost,
        )


class OpenRouterTopProvider(BaseModel):
    context_length: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    is_moderated: bool


class OpenRouterModel(BaseModel):
    id: str
    name: str
    created: int
    description: str
    context_length: int
    architecture: OpenRouterArchitecture
    pricing: OpenRouterModelPricing
    top_provider: Optional[OpenRouterTopProvider] = None
    per_request_limits: Optional[Dict[str, Any]] = None


class OpenRouterModelTable(BaseModel):
    data: List[OpenRouterModel]

    def pricing(self, model_id: str) -> OpenRouterModelPricing:
        if not hasattr(self, "_pricing"):
            self._pricing = {model.id: model.pricing for model in self.data}
        return self._pricing[model_id]


@instance
@retry(
    stop=stop_after_attempt(5),
)
async def openrouter_model_table(logger) -> OpenRouterModelTable:
    async with httpx.AsyncClient() as client:
        response = await client.get("https://openrouter.ai/api/v1/models")
        response.raise_for_status()
        data = response.json()["data"]
        return OpenRouterModelTable.model_validate(dict(data=data))


@instance
def openrouter_api(openrouter_api_key: str):
    return AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1", api_key=openrouter_api_key
    )


@instance
def openrouter_state():
    return dict()


@instance
def openrouter_timeout_sec() -> float:
    return 120


@injected
async def a_openrouter_post(
        openrouter_api_key: str,
        openrouter_timeout_sec: float,
        /,
        payload: dict
) -> dict:
    async with httpx.AsyncClient() as client:
        headers = {
            "Authorization": f"Bearer {openrouter_api_key}",
            'Content-Type': 'application/json',
        }
        response = await client.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload,
                                     timeout=openrouter_timeout_sec)
        return response.json()


@async_cached(sqlite_dict(injected('cache_root_path') / "schema_examples.sqlite"))
@injected
async def a_cached_schema_example_provider(
        a_llm_for_json_schema_example,
        /,
        model_schema: dict
):
    prompt = f"""
    Provide example json objects that follows the schema of the model:{model_schema}
    Beware the example must not be in yaml format.
    If the model contains a list property, provide an example of a case where the list is empty and another example where the list contains multiple items.
    Beware that `type` field is required in the schema, so make sure to include it in the example.
    """
    return await a_llm_for_json_schema_example(prompt)


class OpenRouterChatCompletion(Protocol):
    async def __call__(
            self,
            prompt: str,
            model: str,
            max_tokens: int = 8192,
            temperature: float = 1,
            images: List[PIL.Image.Image] = None,
            response_format: Optional[BaseModel] = None,
            provider: Optional[Dict[str, Any]] = None,
            **kwargs
    ) -> Any:
        ...


@injected
async def a_openrouter_chat_completion__without_fix(
        a_openrouter_post,
        logger,
        openrouter_model_table: OpenRouterModelTable,
        openrouter_state: dict,
        /,
        prompt: str,
        model: str,
        max_tokens: int = 8192,
        temperature: float = 1,
        images: list[PIL.Image.Image] = None,
        response_format=None,
        provider: dict = None,
        **kwargs
):
    """
    :param prompt:
    :param model:
    :param max_tokens:
    :param temperature:
    :param images:
    :param response_format:
    :param provider:  see:https://openrouter.ai/docs/features/provider-routing
    Example:
    provider={'order': [
        'openai',
        'together'
      ],
      allow_fallbacks=False # if everything in order fails, fails the completion. Default is True so some other provider will be used.
    }
    :param kwargs:
    :return:
    """
    provider_filter = dict()
    if response_format is not None and issubclass(response_format, BaseModel):
        provider_filter['provider'] = {
            "require_parameters": True
        }
        openai_response_format = build_openai_response_format(response_format)
        provider_filter['response_format'] = openai_response_format
    if provider is not None:
        p = provider_filter.get('provider', dict())
        p.update(provider)
        provider_filter['provider'] = p

    payload: Dict[str, Any] = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    # *[{"type": "image", "data": img} for img in images or []]
                    *[to_content(img) for img in images or []]
                ]
            }
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        **provider_filter,
        **kwargs
    }
    from pprint import pformat
    res = await a_openrouter_post(payload)
    if 'error' in res:
        raise RuntimeError(f"Error in response: {pformat(res)}")
    cost_dict = openrouter_model_table.pricing(model).calc_cost_dict(res['usage'])
    openrouter_state['cumulative_cost'] = openrouter_state.get('cumulative_cost', 0) + sum(cost_dict.values())

    logger.info(
        f"Cost of completion: {cost_dict}, cumulative cost: {openrouter_state['cumulative_cost']} from {res['provider']}")
    data = res['choices'][0]['message']['content']

    if response_format is not None and issubclass(response_format, BaseModel):
        try:
            if '```' in data:
                data = data.split('```')[1].strip()
            return response_format.model_validate_json(data)
        except Exception as e:
            logger.warning(f"Error in response validation:\n{pformat(payload)}\n{pformat(res)} \n {e} cause:\n{data}")
            data_dict = json_repair.loads(data)
            return response_format.model_validate(data_dict)
    else:
        return data


def build_openai_response_format(response_format):
    pydantic_schema = response_format.model_json_schema()
    pydantic_schema['additionalProperties'] = False
    schema_dict = dict(
        name=response_format.__name__,
        description=f"Pydantic model for {response_format}",
        strict=True,
        schema=pydantic_schema
    )
    openai_response_format = dict(
        type='json_schema',
        json_schema=schema_dict
    )
    return openai_response_format


@injected
async def a_openrouter_chat_completion(
        a_openrouter_post,
        logger,
        a_cached_schema_example_provider: Callable[[type], Awaitable[str]],
        a_structured_llm_for_json_fix,
        openrouter_model_table: OpenRouterModelTable,
        openrouter_state: dict,
        /,
        prompt: str,
        model: str,
        max_tokens: int = 8192,
        temperature: float = 1,
        images: list[PIL.Image.Image] = None,
        response_format=None,
        provider: dict = None,
        **kwargs
):
    """
    :param prompt:
    :param model:
    :param max_tokens:
    :param temperature:
    :param images:
    :param response_format:
    :param provider:  see:https://openrouter.ai/docs/features/provider-routing
    Example:
    provider={'order': [
        'openai',
        'together'
      ],
      allow_fallbacks=False # if everything in order fails, fails the completion. Default is True so some other provider will be used.
    }
    :param kwargs:
    :return:
    """
    provider_filter = dict()
    if response_format is not None and issubclass(response_format, BaseModel):
        provider_filter['provider'] = {
            "require_parameters": True
        }
        openai_response_format = build_openai_response_format(response_format)
        provider_filter['response_format'] = openai_response_format
        schema_prompt = await a_cached_schema_example_provider(response_format.model_json_schema())
        prompt += f"""The response must follow the following json format example:{schema_prompt}"""
    if provider is not None:
        p = provider_filter.get('provider', dict())
        p.update(provider)
        provider_filter['provider'] = p

    payload: Dict[str, Any] = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    # *[{"type": "image", "data": img} for img in images or []]
                    *[to_content(img) for img in images or []]
                ]
            }
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        **provider_filter,
        **kwargs
    }
    from pprint import pformat
    # logger.debug(f"payload:{pformat(payload)}")
    res = await a_openrouter_post(payload)
    if 'error' in res:
        raise RuntimeError(f"Error in response: {pformat(res)}")
    cost_dict = openrouter_model_table.pricing(model).calc_cost_dict(res['usage'])
    openrouter_state['cumulative_cost'] = openrouter_state.get('cumulative_cost', 0) + sum(cost_dict.values())
    # logger.debug(f"response:{pformat(res)}")

    logger.info(
        f"Cost of completion: {cost_dict}, cumulative cost: {openrouter_state['cumulative_cost']} from {res['provider']}")
    data = res['choices'][0]['message']['content']

    if response_format is not None and issubclass(response_format, BaseModel):
        try:
            if '```' in data:
                data = data.split('```')[1].strip()
            return response_format.model_validate_json(data)
        except Exception as e:
            logger.warning(f"Error in response validation:\n{pformat(payload)}\n{pformat(res)} \n {e} cause:\n{data}")
            try:
                data_dict = json_repair.loads(data)
                return response_format.model_validate(data_dict)
            except Exception as e:
                logger.warning(f"json_repair could not repair.{data}")
                fix_prompt = f"""
Please fix the following json object to match the schema:
{data}
                """
                return await a_structured_llm_for_json_fix(fix_prompt, response_format=response_format)
    else:
        return data


@injected
async def a_llm__openrouter(
        openrouter_model_table: OpenRouterModelTable,
        openrouter_api,
        a_openai_compatible_llm,
        logger,
        openrouter_state: dict,
        /,
        text: str,
        model: str,
        response_format=None,
        **kwargs
):
    res: ChatCompletion = await a_openai_compatible_llm(
        api=openrouter_api,
        model=model,
        text=text,
        response_format=response_format,
        **kwargs,
    )

    cost = openrouter_model_table.pricing(model).calc_cost(res.usage)
    total_cost = sum(cost.values())
    openrouter_state['cumulative_cost'] = openrouter_state.get('cumulative_cost', 0) + total_cost
    logger.info(
        f"Cost of completion: {cost}, total cost: {total_cost}, cumulative cost: {openrouter_state['cumulative_cost']}")

    data = res.choices[0].message.content
    if response_format is not None and issubclass(response_format, BaseModel):
        if '```' in data:
            data = data.split('```')[1].strip()
        data = response_format.model_validate_json(data)
    return data


class Text(BaseModel):
    text_lines: list[str]


test_call_gpt4o: IProxy = a_openrouter_chat_completion__without_fix(
    prompt="What is the capital of Japan?",
    model="openai/gpt-4o"
)

test_openai_compatible_llm: IProxy = a_openai_compatible_llm(
    api=openrouter_api,
    model="deepseek/deepseek-chat",
    text="What is the capital of Japan?",
)

test_openrouter_text: IProxy = a_llm__openrouter(
    "What is the capital of Japan?",
    "deepseek/deepseek-chat"
)

test_openrouter_structure: IProxy = a_llm__openrouter(
    f"What is the capital of Japan?.{Text.model_json_schema()}",
    # "deepseek/deepseek-chat",
    "deepseek/deepseek-r1-distill-qwen-32b",
    response_format=Text
)

test_openrouter_model_table: IProxy = openrouter_model_table

test_openrouter_chat_completion: IProxy = a_openrouter_chat_completion(
    prompt="What is the capital of Japan?",
    model="deepseek/deepseek-chat"
)

test_openrouter_chat_completion_with_structure: IProxy = a_openrouter_chat_completion(
    prompt=f"What is the capital of Japan?",
    model="deepseek/deepseek-chat",
    # model="deepseek/deepseek-r1-distill-qwen-32b",
    response_format=Text
)

test_return_empty_item: IProxy = a_openrouter_chat_completion(
    prompt=f"Please answer with empty lines.",
    model="deepseek/deepseek-chat",
    response_format=Text
)

@instance
def __debug_design():
    from openrouter.instances import a_cached_sllm_gpt4o__openrouter
    from openrouter.instances import a_cached_sllm_gpt4o_mini__openrouter
    return design(
        a_llm_for_json_schema_example=a_cached_sllm_gpt4o__openrouter,
        a_structured_llm_for_json_fix=a_cached_sllm_gpt4o_mini__openrouter,
    )

__meta_design__ = design(
    overrides=__debug_design
)
