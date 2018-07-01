from django.shortcuts import render
from rest_framework.parsers import JSONParser
from rest_framework.renderers import JSONRenderer
from django.http import JsonResponse
from .serializers import InputImagerSerializer
from django.views.decorators.http import require_http_methods
# Create your views here.
from django.views.decorators.csrf import csrf_exempt
from django.views import View
from django.utils.decorators import method_decorator
import numpy as np
from PIL import Image
import io
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

model = load_model('doggo_classifier_2')
graph = tf.get_default_graph()

@method_decorator(csrf_exempt, name='dispatch')
class PredictView(View):
    def post(self, request):

        image = Image.open(io.BytesIO(request.body))
        image = image.resize((100, 100))
        image = np.array(image).transpose(1, 0, 2)
        image = image.reshape((-1, image.shape[0], image.shape[1], image.shape[2]))


        image_gen = ImageDataGenerator(rescale=1.0 / 255)
        image_gen = image_gen.flow(image, )

        breeds = {'beagle': 0,
                  'boxer': 1,
                  'bull_mastiff': 2,
                  'doberman': 3,
                  'german_shepherd': 4,
                  'golden_retriever': 5,
                  'labrador_retriever': 6,
                  'pomeranian': 7,
                  'pug': 8,
                  'rottweiler': 9}
        test_image = image_gen.next()


        with graph.as_default():
            percentiles = model.predict(test_image)
            image_class = model.predict_classes(test_image)

        result = {"percentiles":{},"predicted_class":""}
        for b in breeds.keys():
            result["percentiles"][b] = str(percentiles[0][breeds[b]])
            if breeds[b] == image_class[0]:
                result["predicted_class"] = b
        print(result)
        return JsonResponse(result, status=200)
