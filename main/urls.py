from django.urls import path
from django.contrib.auth.views import LoginView, LogoutView

from . import views

urlpatterns = [
    path("", views.videolist, name="index"),
    path("photoview", views.photoview, name="photoview"),
    path("allpackges", views.allpackges, name="allpackges"),
    path("packege1", views.packege1, name="packege1"),
    path("packege2", views.packege2, name="packege2"),
    path("packege3", views.packege3, name="packege3"),
    #path("image", views.image, name="image"),
    path("useritemstore", views.create_user_item, name="createuser"),
    path('useritemshow', views.retrieve_user_items, name='user_items'),
    path('accounts/logout/', LogoutView.as_view(), name='logout'),
    path(
        'accounts/login/',
        LoginView.as_view(template_name='login.html'),
        name='login'
    ),
    path('upload/', views.upload_photos, name='upload_photos'),
    path('proxy', views.proxy_view, name='proxy'),
    

    #new 
    path('uploadvideo/', views.upload_video, name='upload_video'),
    path('videolist/', views.videolist, name='videolist'),
    path('upload/success/', views.upload_success, name='upload_success'),



    path('convert/', views.convert_video_to_audio, name='convert_video_to_audio'),
    path('youtube_search/<str:query>/', views.youtube_search, name='youtube_search'),

]