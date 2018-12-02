let http = require('http');
let path = require('path');
let fs = require('fs');
let mimeTypes = {
        '.ply' : 'text/plain',
        '.obj' : 'text/plain',
        '.vs' : 'text/plain',
        '.frag' : 'text/plain',
        '.js' : 'text/javascript',
        '.html' : 'text/html',
        '.css' : 'text/css'
    };

function handleRequest(request, response) {
    let lookup = (request.url === '/') ? '/index.html' : decodeURI(request.url);
    let file = lookup.substring(1, lookup.length);
    console.log('request: ' + request.url);

    fs.exists(file, function(exists) {
        console.log(exists ? lookup + ' is there' : lookup + ' doesn\'t exist');
        if (exists) {
            fs.readFile(file, function(err, data) {
                if (err) {
                    response.writeHead(500);
                    response.end('Server Error!');
                } else {
                    let headers = {'Content-type': mimeTypes[path.extname(lookup)]};
                    response.writeHead(200, headers);
                    response.end(data);
                }
            });
        } else {
            response.writeHead(404);
            response.end();
        }
    });
}

http.createServer(handleRequest).listen(3000, function() {
    console.log('Server is listening on port 3000: ');
});
