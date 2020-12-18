from django.http import HttpResponse, JsonResponse
from ..modelart.customize_service import ObjectDetectionService

detect_service = ObjectDetectionService()


def detect(request):
    data = {str(0): {'file_name': '/home/lifeng/data/detection/fabric/trainval/e16b32a11e9ef68e0841094813.jpg'}}
    result = detect_service.inference2(data=data)
    return JsonResponse(result)
