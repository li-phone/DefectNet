from django.http import HttpResponse, JsonResponse
from ..modelart.customize_service import ObjectDetectionService
from django.shortcuts import render

detect_service = ObjectDetectionService()


def detect(request):
    if request.method == 'POST':  # 获取对象
        obj = request.FILES.get('image')
        data = {obj.name: {'file_name': obj}}
        result = detect_service.inference2(data=data)
    else:
        data = {str(0): {'file_name': '/home/lifeng/data/detection/fabric/trainval/e16b32a11e9ef68e0841094813.jpg'}}
        result = detect_service.inference2(data=data)
        result['message'] = 'Demo test result!'
    return JsonResponse(result)
