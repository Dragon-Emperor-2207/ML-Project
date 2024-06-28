from django.urls import path
from .views import *

urlpatterns = [
    path('login/', loginUser, name='Login'),
    path('logout/', logoutUser, name='Logout'),
    path('register/', registerUser, name='Register')
]