import rest_framework.serializers as serializers


class InputImagerSerializer(serializers.Serializer):
    data = serializers.CharField(required=True)

