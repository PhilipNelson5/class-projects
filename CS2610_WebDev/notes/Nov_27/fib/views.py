# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.http import HttpResponse
from django.shortcuts import render
import json

def fibonacci(n):
	if (n == 0):
		return 0
	elif (n == 1):
		return 1
	else:
		return fibonacci(n-1) + fibonacci(n-2)


def fibAPI(request):
	for k, v in request.GET.items():   # DELETE ME
		print k, "=>", v   # DELETE ME

	resp = {}

	if not request.GET.has_key('n'):
		resp['error'] = "Usage: n=[non-negative integer]"
	else:
		n = int(request.GET['n'])
		if (n < 0):
			resp['error'] = "Usage: n=[non-negative integer]"
		else:
			val = fibonacci(n)
			resp = { 'n': n, 'fibonacci': val }

	return HttpResponse(json.dumps(resp))

def index(request):
    return render(request, 'fib/index.html', {})
