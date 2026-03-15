from django.shortcuts import render

# Create your views here.
from rest_framework.decorators import api_view
from rest_framework.response import Response

from translator.inference import translate_code


@api_view(['POST'])
def translate(request):
    
    source_lang = request.data.get("source_lang")
    target_lang = request.data.get("target_lang")
    code = request.data.get("code")

    if not source_lang or not target_lang or not code:
        return Response({"error": "source_lang, target_lang, and code are required"}, status=400)

    result = translate_code(source_lang, target_lang, code)

    return Response({"translated_code": result})
