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

//------------------------------------------------------------------
//
// Helper function used to normalize an N dimensional vector
//
//------------------------------------------------------------------
  function normalize(array) {
    const mag = array.reduce((acc, val)=>{return acc + val;}, 0.0);
    return array.map(val => val/mag);
  }

