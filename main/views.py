from django.shortcuts import render
from rest_framework.decorators import api_view
# Create your views here.
from google.cloud import speech_v1p1beta1 as speech
from django.contrib.auth.decorators import login_required
import requests
import json
from .models import UserItem, Userprofile,Photo
from django.http import HttpResponse
from rest_framework.response import Response
from django.shortcuts import render, redirect
from .forms import PhotoUploadForm
from io import BytesIO
from django.http import JsonResponse
from PIL import Image, ImageOps
import tempfile
from django.core.files.uploadedfile import InMemoryUploadedFile
from django.shortcuts import get_object_or_404
import uuid
import requests
import json
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
from django.shortcuts import render, redirect
from .models import Video
from .forms import VideoUploadForm
import re
from django.core.files.base import ContentFile
import moviepy.editor as mp
import speech_recognition as sr
from moviepy.editor import VideoFileClip
from django.core.files import File
import os
import io
from django.http import FileResponse
import gensim
from gensim import corpora
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from django.conf import settings

BASE_DIR = settings.BASE_DIR




@login_required
def index(request):
    user = request.user
    user_item =  Userprofile.objects.filter(user=user).first()  # Retrieve the prfile object related to the current user
    context = {
        'user': user,
        'modelid': user_item.modelid if user_item else None,
        'image': user_item.image if user_item else None,
    }


    return render(request, 'form.html',context)


@login_required
def photoview(request):
    user = request.user
    user_item = Userprofile.objects.filter(user=user).order_by('-id').first() # Retrieve the UserItem object related to the current user
    context = {
        'user': user,
        'modelid': user_item.modelid if user_item else None,
        'image': user_item.image if user_item else None,
    }


    return render(request, 'a.html',context)


@api_view(['POST'])
def create_user_item(request):
    data = request.data
    user = request.user
    modelid = data.get('modelid')
    image = data.get('image')
    
    # Create the UserItem object
    user_item = UserItem.objects.create(user=user, modelid=modelid, image=image)

    # Return a success response
    return Response({'message': 'success'})



@login_required
def image(request):
 

            url = "https://stablediffusionapi.com/api/v4/dreambooth"

            payload = json.dumps({
            "key": "qcAvUL9oteJVVsX8I8P2986GxkoSqGvTmejDQHpdMJChUKrVTnXJrcYVn4Rp",
    "model_id": "H6T5co3VwC0dJeBuLpXsWDhSG",
      
    
     "prompt": "abir_rohan1811 person,ed:1.15),shiny glitter party dress,whole full body,detailed high end fashion,formal dress,natural light, sharp, detailed face, magazine, photo, canon, nikon, focus, award winning photo,reminiscent of the works of Steve McCurry, 35mm, F/2.8, insanely detailed and intricate, character, hypermaximalist, elegant, ornate, hyper realistic, super detailed, trending on flickr, portrait photo,masterpiece,beach background,full body, best quality, high resolution, 8K , HDR, bloom, sun light, raytracing , detailed shadows, intricate tree shadow, bokeh, depth of field, film photography, film grain, glare, (wind:0.8), detailed hair, beautiful face, beautiful man, ultra detailed eyes, cinematic lighting, (hyperdetailed:1.15), outdoors,happy face,,ultra-realistic,clear facial features,natural features,captured with a phase one 35mm lens,f/ 3.2,agfa vista film ,film grain light,global illumination,intricate detail,wide shot,--upbeta --ar 4:5 --s 750 --q 2", 
  "negative_prompt": "(worst quality:2.00), (low quality:2.00), (normal quality:2.00), low-res,deformed face,(deformed iris, deformed pupils, semi-realistic, CGI, 3D, render, sketch, cartoon, drawing, anime:1.4), text, close up, cropped, out of frame, worst quality, low quality, JPEG artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck,blurry background,dslr,portrait,dark face,dark eye,make up,fat face, easynegative,flat face,red skin,bad skin, wrinkles, pustules",
            "width": "512",
            "height": "512",
            "samples": "1",
            "num_inference_steps": "30",
            "safety_checker": "no",
            "enhance_prompt": "yes",
            "seed": None,
            "guidance_scale": 7.5,
            "multi_lingual": "no",
            "panorama": "no",
            "self_attention": "no",
            "upscale": "no",
            "embeddings_model": None,
            "lora_model": None,
            "scheduler": "UniPCMultistepScheduler",
            "webhook": None,
            "track_id": None
            })

            headers = {
            'Content-Type': 'application/json'
            }

            response = requests.request("POST", url, headers=headers, data=payload)

            print(response.text)
            return render(request, 'a.html')



def retrieve_user_items(request):
    user = request.user
    user_items = UserItem.objects.filter(user=user).order_by('-id')   
    data = {
        'user_items': list(user_items.values())
    }



    user_items = Photo.objects.filter(user=request.user).order_by('-id')
    image_urls = [item.image.url for item in user_items]

# Convert image URLs to the desired format
    formatted_images = []
    for url in image_urls:
        formatted_images.append(url)   


    print(formatted_images)
    

# If you want to add the "images" key to the output, you can create a dictionary
   


    return JsonResponse(data)




def resize_image(image, username, max_size=(512, 512)):
    # Open the image using PIL to get an Image object that supports getexif()
    img = Image.open(image)

    # Rotate the image according to the Exif orientation tag
    img = ImageOps.exif_transpose(img)

    # Calculate the new size while maintaining the aspect ratio
    width, height = img.size
    max_dim = max(width, height)

    # Calculate the new dimensions
    new_width = int(max_size[0] * width / max_dim)
    new_height = int(max_size[1] * height / max_dim)

    # Resize the image without rotating
    img = img.resize((new_width, new_height), Image.LANCZOS)

    # Create a new BytesIO object to store the resized image data
    output_io = BytesIO()

    # Save the resized image to the BytesIO object in PNG format
    img.save(output_io, format='png')

    # Generate a unique identifier
    unique_id = uuid.uuid4().hex

    # Concatenate the username and the unique identifier to form the new filename
    new_filename = f"{username}_{unique_id}.png"

    # Create an InMemoryUploadedFile from the BytesIO data with the new filename
    image_file = InMemoryUploadedFile(
        output_io, None, new_filename, 'image/png', output_io.getbuffer().nbytes, None
    )

    return image_file



def upload_photos(request):
    if request.method == 'POST':
        images = request.FILES.getlist('images')
        selected_gender = request.POST.get('gender') 
        username = request.user.username

        for image in images:
            # Resize the image before creating the Photo object
            resized_image = resize_image(image, username)

            # Create a Photo object without saving it to the database yet
            photo = Photo(
                user=request.user,
                image=resized_image,
            )

            # Save the Photo object to the database
            photo.save()

        user_items = Photo.objects.filter(user=request.user).order_by('-id')
        image_urls = [item.image.url for item in user_items]

# Convert image URLs to the desired format
        formatted_images = []
        for url in image_urls:
            formatted_images.append(url)    


        

        url = "https://stablediffusionapi.com/api/v3/fine_tune_v2"

        payload = json.dumps({
        "key": "KhwsDivKGA4dhHDlbZQcVeGojvjTjIMNCebIV6iImfvgWPHpyW0E8ZWAjK33",
        "instance_prompt": "photo of abir_rohan1811",
        "class_prompt": "photo of person",
        "base_model_id": "realistic-vision-v13",
         "images": formatted_images,
        "seed": "0",
  "training_type": selected_gender ,
  "learning_rate_unet": "2e-6",
  "steps_unet": "1500",
  "learning_rate_text_encoder": "1e-6",
  "steps_text_encoder": "350",
  "webhook": ""
})

        headers = {
        'Content-Type': 'application/json'
        }

        
        response = requests.request("POST", url, headers=headers, data=payload)
        response_dict = json.loads(response.text)
        modelid=response_dict["model_id"]

        
       
        
        user_item = get_object_or_404(Userprofile, user=request.user)
        user_item.modelid = modelid
      
# Update the modelid attribute
        

        user_item.save()



    return render(request, 'upload_photos.html')


@csrf_exempt
def proxy_view(request):
   if request.method == 'POST':
        target_url = "https://stablediffusionapi.com/api/v4/dreambooth" # Replace with your target API URL

        try:
            # Forward the incoming request to the target API
            response = requests.post(target_url, data=request.body, headers={'Content-Type': 'application/json'})

            # Get the JSON response from the target API
            response_data = response.json()

            # Send the JSON response back to the client-side application
            return JsonResponse(response_data)

        except requests.exceptions.JSONDecodeError as e:
            # If there was an error parsing the JSON response from the target API
            return JsonResponse({'error': str(e)}, status=500)

   return JsonResponse({'error': 'Invalid request method'}, status=400)







@login_required
def allpackges(request):
    


    return render(request, 'all_pakages.html')





@login_required
def packege1(request):
    user = request.user
    user_item =  Userprofile.objects.filter(user=user).first()  # Retrieve the prfile object related to the current user
    context = {
        'user': user,
        'modelid': user_item.modelid if user_item else None,
        'image': user_item.image if user_item else None,
    }


    return render(request, 'packege1.html',context)



@login_required
def packege2(request):
    user = request.user
    user_item =  Userprofile.objects.filter(user=user).first()  # Retrieve the prfile object related to the current user
    context = {
        'user': user,
        'modelid': user_item.modelid if user_item else None,
        'image': user_item.image if user_item else None,
    }


    return render(request, 'packege2.html',context)


@login_required
def packege3(request):
    user = request.user
    user_item =  Userprofile.objects.filter(user=user).first()  # Retrieve the prfile object related to the current user
    context = {
        'user': user,
        'modelid': user_item.modelid if user_item else None,
        'image': user_item.image if user_item else None,
    }


    return render(request, 'packege3.html',context)







def upload_success(request):
    return render(request, 'upload_success.html')



def upload_video(request):
    if request.method == 'POST':
        form = VideoUploadForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()

            # Save the transcription result in the video instance
            #transcribe_video.delay(video.id)

            # Convert the uploaded video to MP3
            

            return redirect('upload_success')
    else:
        form = VideoUploadForm()
    return render(request, 'upload.html', {'form': form})




# ... (previous code)

def convert_video_to_audio(request):
    if request.method == 'POST' and request.FILES.get('video'):
        video_file = request.FILES['video']
        audio_path = 'converted_audio.mp3'  # Path to store the converted audio

        video_clip = VideoFileClip(video_file.temporary_file_path())
        audio_clip = video_clip.audio

        # Specify the target audio codec and format
        audio_codec = 'mp3'
        audio_format = 'mp3'
        audio_clip.write_audiofile(audio_path, codec=audio_codec, fps=audio_clip.fps, write_logfile=True)

        # Transcribe audio using Google Cloud Speech-to-Text
        client = speech.SpeechClient()
        with open(audio_path, 'rb') as audio_file:
            content = audio_file.read()

        audio = speech.RecognitionAudio(content=content)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.MP3,  # Specify MP3 encoding
            sample_rate_hertz=16000,
            language_code="en-US",
        )

        response = client.recognize(config=config, audio=audio)
        transcription_text = ''
        for result in response.results:
            transcription_text += result.alternatives[0].transcript + ' '

        stop_words = set(stopwords.words('english'))
        transcription_tokens = [word for word in word_tokenize(transcription_text.lower()) if word not in stop_words]

        # Create a dictionary representation of the transcribed tokens.
        transcription_dictionary = corpora.Dictionary([transcription_tokens])

        # Convert the dictionary into a document-term matrix.
        transcription_corpus = [transcription_dictionary.doc2bow(transcription_tokens)]

        # Generate the LDA model for the transcription.
        ldamodel_transcription = gensim.models.ldamodel.LdaModel(transcription_corpus, num_topics=15, id2word=transcription_dictionary, passes=15)

        topics_transcription = ldamodel_transcription.print_topics(num_words=6)



        #experiment

        topics_with_words = []
        word_pattern = re.compile(r'"([^"]+)"')

        for entry in topics_transcription:
            words = word_pattern.findall(entry[1])
            topics_with_words.append((entry[0], words))
        

        # Prepare data to pass to the template
        context = {
            'topics_transcription': topics_with_words,
            'transcription_text': transcription_text,
            
        }

        # Render the template with the context data
        return render(request, 'transcription_result.html', context)

    return render(request, 'convert.html')



# def convert_video_to_audio(request):
#     if request.method == 'POST' and request.FILES.get('video'):
#         video_file = request.FILES['video']

#         try:
#             # Initialize Google Cloud Storage client
#             gs_credentials = settings.GOOGLE_CLOUD_CREDENTIALS
#             gs_bucket_name = settings.GOOGLE_CLOUD_STORAGE_BUCKET
#             gs_project_id = settings.GOOGLE_CLOUD_PROJECT_ID

#             storage_client = storage.Client(project=gs_project_id, credentials=gs_credentials)
#             bucket = storage_client.bucket(gs_bucket_name)

#             # Upload the video file to Google Cloud Storage
#             blob = bucket.blob('video.mp4')
#             blob.upload_from_file(video_file.temporary_file_path())

#             # Initialize Google Cloud Speech client
#             speech_client = speech.SpeechClient(credentials=gs_credentials)

#             # Specify the GCS URI for the audio file
#             audio_uri = f'gs://{bucket.name}/video.mp4'

#             # Configure the recognition request
#             audio = types.RecognitionAudio(uri=audio_uri)
#             config = types.RecognitionConfig(
#                 encoding=enums.RecognitionConfig.AudioEncoding.LINEAR16,
#                 sample_rate_hertz=16000,
#                 language_code='en-US',
#             )

#             operation = speech_client.long_running_recognize(config=config, audio=audio)

#             # Wait for the operation to complete
#             response = operation.result()

#             # Extract the transcription text
#             transcription_text = ''
#             for result in response.results:
#                 transcription_text += result.alternatives[0].transcript + ' '

#             # Process and analyze the transcription text as needed

#             # Prepare data to pass to the template
#             context = {
#                 'transcription_text': transcription_text,
#             }

#             # Render the template with the context data
#             return render(request, 'transcription_result.html', context)

#         except Exception as e:
#             return HttpResponse(f"An error occurred: {str(e)}")

#     return render(request, 'convert.html')







@login_required
def videolist(request):
    
    videos = Video.objects.all()
    return render(request, 'videolist.html', {'videos': videos})



def transcribe_video(video_id):
    video = Video.objects.get(pk=video_id)
    model = whisper.load_model("base")
    result = model.transcribe(video.video_file.path)

    print(result['text'])
    video.save()  



def youtube_search(request, query):
    api_key = settings.YOUTUBE_API_KEY
    api_url = settings.YOUTUBE_API_URL
    params = {
        'key': api_key,
        'q': query,
        'part': 'snippet',
        'maxResults': 10,  # Adjust the number of results as needed
    }

    response = requests.get(api_url, params=params)
    data = response.json()

    context = {
        'search_results': data.get('items', []),
    }
 ## jjjj
    return render(request, 'youtube_search_results.html', context) 

  

    















