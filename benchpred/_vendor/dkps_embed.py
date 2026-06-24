"""Embedding API (Google-only) — vendored & slimmed from DKPS.

Trimmed copy of the Google provider path from the DKPS project's ``dkps/embed.py``
(plus the ``disk_cache`` helper from ``dkps/cache.py``), kept here so benchpred
does not depend on the external ``dkps`` package.

Provenance:
    source : https://github.com/edlwang/dkps  (dkps/embed.py, dkps/cache.py)
    commit : 5a84ea8945ba329d4157e57433d74fbc5980f15d

Changes vs upstream:
    * Only the ``google`` provider is kept (jina / openrouter / litellm /
      huggingface / sentence-transformers / jlai paths removed), so the only
      extra runtime dependency is ``google-genai``.
    * ``disk_cache`` uses plain ``print`` instead of ``rich`` (one fewer dep).
    * ``httpx`` import dropped (it was only used by the Jina path).

Requires ``GEMINI_API_KEY`` in the environment.  Embeddings are disk-cached
under ``$DKPS_CACHE_DIR/embed/google`` (default ``./.cache/embed/google``),
keyed by the chunk contents + model, so repeated runs reuse them.
"""

import os
import inspect
import pickle
import hashlib
import asyncio
from functools import wraps

import numpy as np
from tqdm.asyncio import tqdm

try:
    from google import genai
    from google.genai.types import HttpOptions
except Exception:  # pragma: no cover - exercised only without google-genai installed
    genai = None
    HttpOptions = None


# Base directory for on-disk embedding caches.  Override with DKPS_CACHE_DIR.
# Resolved at import time, so set it before importing this module.
_CACHE_BASE = os.path.join(os.environ.get('DKPS_CACHE_DIR', './.cache'), 'embed')


def disk_cache(cache_dir='./.cache', verbose=False, ignore_fields=None):
    """Cache a (sync or async) function's results to disk by argument hash."""
    os.makedirs(cache_dir, exist_ok=True)

    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            cache_str, cache_path = _get_cache_info(func, args, kwargs)
            cached_result = _try_get_cached_result(cache_path, cache_str, verbose)
            if cached_result is not None:
                return cached_result
            result = await func(*args, **kwargs)
            _save_to_cache(result, cache_path, cache_str, verbose)
            return result

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            cache_str, cache_path = _get_cache_info(func, args, kwargs)
            cached_result = _try_get_cached_result(cache_path, cache_str, verbose)
            if cached_result is not None:
                return cached_result
            result = func(*args, **kwargs)
            _save_to_cache(result, cache_path, cache_str, verbose)
            return result

        def _get_cache_info(func, args, kwargs):
            sig = inspect.signature(func)
            params = {}
            for param_name, param in sig.parameters.items():
                if param.default is not param.empty:
                    params[param_name] = param.default
            positional_params = list(sig.parameters.keys())
            for i, arg in enumerate(args):
                if i < len(positional_params):
                    params[positional_params[i]] = arg
            params.update(kwargs)
            if ignore_fields:
                for field in ignore_fields:
                    if field in params:
                        del params[field]
            cache_str = '-> '.join([func.__name__, str(sorted(params.items()))])
            cache_key = hashlib.md5(''.join(cache_str).encode()).hexdigest()
            cache_path = os.path.join(cache_dir, f"{cache_key}.pkl")
            return cache_str, cache_path

        def _try_get_cached_result(cache_path, cache_str, verbose):
            if os.path.exists(cache_path):
                try:
                    out = pickle.load(open(cache_path, 'rb'))
                    if verbose:
                        print(f"disk_cache: Loaded from cache {cache_path}")
                    return out
                except Exception as e:
                    print(f"disk_cache: Error loading cache: {cache_dir} {cache_path} {e}")
            elif verbose:
                print(f"disk_cache: No cache found {cache_dir} {cache_path} - Running")
            return None

        def _save_to_cache(result, cache_path, cache_str, verbose):
            try:
                with open(cache_path, 'wb') as f:
                    pickle.dump(result, f)
            except Exception as e:
                print(f"disk_cache: Error saving to cache: {cache_str} {e}")

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


@disk_cache(cache_dir=os.path.join(_CACHE_BASE, 'google'), verbose=False, ignore_fields=['client'])
async def _aembed_google_chunk(chunk_id, client, chunk, model):
    chunk_response = await client.aio.models.embed_content(
        model=model,
        contents=chunk,
    )
    return chunk_id, np.array([xx.values for xx in chunk_response.embeddings])


async def _aembed_api(provider, input_strs, chunk_size=50, max_concurrency=5, model=None):
    """Async embedding with chunking + bounded concurrency.  Google only."""
    assert isinstance(input_strs, list), 'input_strs must be a list'
    if provider != 'google':
        raise ValueError(
            f"Unsupported provider {provider!r}; only 'google' is vendored. "
            "Use the upstream dkps package for other providers."
        )
    if genai is None:
        raise ImportError(
            "google-genai is not installed. Install the optional extra: "
            "pip install -e .[dkps]"
        )

    client = genai.Client(
        api_key=os.getenv('GEMINI_API_KEY'),
        http_options=HttpOptions(api_version='v1beta', timeout=10 * 1000),
    )
    _aembed_chunk = _aembed_google_chunk
    if model is None:
        model = 'gemini-embedding-001'

    sem = asyncio.Semaphore(max_concurrency)

    async def _fn(chunk_id, chunk):
        async with sem:
            return await _aembed_chunk(chunk_id, client, chunk, model)

    chunks = [input_strs[i:i + chunk_size] for i in range(0, len(input_strs), chunk_size)]
    tasks = [_fn(chunk_id, chunk) for chunk_id, chunk in enumerate(chunks)]

    out = [None] * len(chunks)
    for task in tqdm(asyncio.as_completed(tasks), desc="Embedding chunks", total=len(tasks)):
        chunk_id, embedding = await task
        out[chunk_id] = embedding

    return np.concatenate(out)


def embed_api(provider, *args, **kwargs):
    """Synchronous wrapper around the async Google embedder."""
    return asyncio.run(_aembed_api(provider, *args, **kwargs))
