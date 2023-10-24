# Generated by Django 4.2.3 on 2023-08-16 11:00

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("main", "0003_photo"),
    ]

    operations = [
        migrations.CreateModel(
            name="Video",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("title", models.CharField(max_length=100)),
                ("video_file", models.FileField(upload_to="videos/")),
                ("uploaded_at", models.DateTimeField(auto_now_add=True)),
            ],
        ),
    ]