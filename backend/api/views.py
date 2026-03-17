from django.shortcuts import render
from rest_framework.decorators import api_view , throttle_classes  # add throttle_classes here
from rest_framework.response import Response
from rest_framework.throttling import UserRateThrottle
from translator.inference import translate_code
from asyncio import Semaphore
import asyncio
import logging

# -------------------------------
# GPU Semaphore: only 1 translation at a time
# -------------------------------
gpu_semaphore = Semaphore(1)

# Configure logging
logger = logging.getLogger("translator")
logger.setLevel(logging.INFO)

class TranslateRateThrottle(UserRateThrottle):
    rate = '5/minute'   #specific for this endpoint

@api_view(['POST'])
@throttle_classes([TranslateRateThrottle])
async def translate(request):
    """
    POST body: {
        "source_lang": "python",
        "target_lang": "cpp",
        "code": "...",
        "model_name": "incoder"  # optional
    }
    """
    source_lang = request.data.get("source_lang")
    target_lang = request.data.get("target_lang")
    code = request.data.get("code")
    model_name = request.data.get("model_name", "incoder")  # Optional model selection

    # Validate inputs
    if not source_lang or not target_lang or not code:
        return Response({"error": "source_lang, target_lang, and code are required"}, status=400)

    try:
        # --------------------------------------------------------------------------
        # Run translation in background thread using per-request model safely on GPU
        # --------------------------------------------------------------------------
        async with gpu_semaphore:
            # run translate_code in a thread so Django async view doesn't block
            translated_code = await asyncio.to_thread(
                translate_code,
                source_lang,
                target_lang,
                code,
                model_name   # <-- pass the model_name from request
            )
    except Exception as e:
        logger.warning(f"Translation failed", exc_info=True)
        return Response({"error": f"Translation failed: {str(e)}"}, status=500)

    return Response({"translated_code": translated_code})