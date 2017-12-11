from django.conf.urls import url

from . import views

app_name = 'fib'
urlpatterns = [
	# /fib/fibAPI?n=[some non-negative number]
    url(r'^fibAPI$', views.fibAPI, name='fibAPI'),

	# /fib/
    url(r'^$', views.index, name='index'),
]
