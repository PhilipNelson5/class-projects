var red = document.querySelector('.red')
var p0 = document.createElement('p')
p0.textContent = "First, take a look at this page's HTML document"
var p1 = document.createElement('p')
p1.textContent = "(It's the file called index.html)"
var p2 = document.createElement('p')
p2.textContent = "Where are all of these words comming from?"
red.appendChild(p0)
red.appendChild(p1)
red.appendChild(p2)


var yellow = document.querySelector('.yellow')
p0 = document.createElement('p')
p0.textContent = "Then take a look at the stylesheet for this page"
p1 = document.createElement('p')
p1.textContent = "(It is a file called 'style.css')"
p2 = document.createElement('p')
p2.textContent = "Did you find what you are looking for?"
yellow.appendChild(p0)
yellow.appendChild(p1)
yellow.appendChild(p2)


p0 = document.createElement('p')
p0.textContent = "Then have a look at that JavaScript file mentioned at the end"
p1 = document.createElement('p')
p1.textContent = "(This file is called 'content.js')"
var blue = document.querySelector('.blue')
blue.appendChild(p0)
blue.appendChild(p1)

p0 = document.createElement('p')
p0.textContent = "Does that finally solve the mystery?"
var green = document.querySelector('.green')
green.appendChild(p0)

