from django.db import models
from django.contrib.auth.models import User
from django.core.files.base import ContentFile
from moviepy.editor import VideoFileClip
import os
# Create your models here.

class Userprofile(models.Model):
    
    user = models.ForeignKey(User, on_delete=models.CASCADE)
   

    modelid = models.CharField(max_length=500,blank=True,null=True)
    image = models.CharField(max_length=500,blank=True,null=True)

class UserItem(models.Model):
    
    user = models.ForeignKey(User, on_delete=models.CASCADE)
   
    userprofile = models.ForeignKey(Userprofile, on_delete=models.CASCADE,null=True,related_name='userprofile')
    modelid = models.CharField(max_length=500,blank=True,null=True)
    image = models.CharField(max_length=500,blank=True,null=True)



class Photo(models.Model):
    user = models.ForeignKey('auth.User', on_delete=models.CASCADE)
    image = models.ImageField(upload_to='photos/')
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username}'s photo at {self.uploaded_at}"   


class Video(models.Model):
    title = models.CharField(max_length=100)
    video_file = models.FileField(upload_to='videos/')
    mp3 = models.FileField(upload_to='mp3/',null =True)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    transcription = models.CharField(max_length=1000000,null= True)
    all_topic = models.CharField(max_length=1000 ,null =True)
    def __str__(self):
        return self.title    

    # def save(self, *args, **kwargs):
    #     super(Video, self).save(*args, **kwargs)

    #     if not self.mp3 and self.video_file:
    #         video_path = self.video_file.path
    #         mp3_filename = os.path.splitext(os.path.basename(video_path))[0] + ".mp3"
    #         mp3_path = os.path.join("mp3", mp3_filename)

    #         try:
    #             video_clip = VideoFileClip(video_path)
    #             audio_clip = video_clip.audio
    #             mp3_content = audio_clip.write_to_buffer()
    #             audio_clip.close()
    #             video_clip.close()

    #             self.mp3.save(mp3_filename, ContentFile(mp3_content), save=False)
    #             self.save(update_fields=["mp3"])

    #         except Exception as e:
    #             print("Error generating MP3:", e)
    #             pass


