'use strict';

//------------------------------------------------------------------
//
// Helper function used to load a file from the server
//
//------------------------------------------------------------------
function loadFileFromServer(filename) {
    return new Promise((resolve, reject) => {
        let xmlHttp = new XMLHttpRequest();

        xmlHttp.onreadystatechange = function() {
            if (xmlHttp.readyState === XMLHttpRequest.DONE) {
                resolve(xmlHttp.responseText);
            }
        };

        xmlHttp.open("GET", filename, true);
        xmlHttp.send();
    });
}
