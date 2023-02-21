from django.http import HttpResponse
from django.shortcuts import render
from .forms import UserForm
from .model.predict import predict


def index(request):
    if request.method == "POST":
        review = request.POST.get("review")
        userform = UserForm()
        sentiment, rating = predict(review)
        result = 'Sentiment: ' + str(sentiment) + ', Rating: ' + str(rating)
        return render(request, "index.html", {"form": userform, "result": result})
    else:
        userform = UserForm()
        return render(request, "index.html", {"form": userform, "result": ''})