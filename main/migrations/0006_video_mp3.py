# Generated by Django 4.2.3 on 2023-08-30 05:55

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("main", "0005_video_all_topic_video_transcription"),
    ]

    operations = [
        migrations.AddField(
            model_name="video",
            name="mp3",
            field=models.FileField(null=True, upload_to="mp3/"),
        ),
    ]