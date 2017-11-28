#!/usr/bin/env python

<form action = "{% url 'blog:post_a_comment' blog_id %}">

def postComment(request, blog_id):
    for k, v in request.POST.items():
        print k, " -> ", v

    c = Comment()
    c.nickname = request.POST.nickname
    c.comment = request.POST.comment
    ...

    b = Blog.objects.get(id=blog_id)
    c.blog = b #c.foreign_key = b
    c.save()
    return HttpRedirect(...)

# Detail view
# List view : Home Page
