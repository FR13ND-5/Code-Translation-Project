from django.shortcuts import render
from .model_inference import translate_code

def home(request):
    output = ""
    code = ""

    if request.method == "POST":
        code = request.POST.get("code")
        output = translate_code(code)

    return render(request, "index.html", {"output": output, "code": code})