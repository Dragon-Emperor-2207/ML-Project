from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.contrib.auth import authenticate, login, logout
from rest_framework import status
from .forms import CustomUserCreationForm

# Create your views here.
@api_view(['POST'])
def loginUser(request):
    username = request.POST['username']
    password = request.POST['password']

    user = authenticate(request, username=username, password=password)

    if user is not None:
        login(request, user)
        return Response({'details': 'User Credentials Match'}, status=status.HTTP_200_OK)
    return Response({'details': "User Credentials don't match"}, status=status.HTTP_404_NOT_FOUND)

@api_view(['POST'])
def logoutUser(request):
    logout(request)
    return Response({'details': 'Logout Successful'}, status=status.HTTP_200_OK)

@api_view(['POST'])
def registerUser(request):
    form = CustomUserCreationForm(request.data)
    if form.is_valid():
        user = form.save()
        login(request, user)
        return Response({"Details":"Registration Success"}, status=status.HTTP_200_OK)
    
    return Response({"details": "Registration unsuccessful", "errors": form.errors}, status=status.HTTP_400_BAD_REQUEST)