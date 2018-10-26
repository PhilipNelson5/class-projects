ModelLoaderPLY = (function() {
    'use strict';


    //------------------------------------------------------------------
    //
    // Placeholder function that returns a hard-coded cube.
    //
    //------------------------------------------------------------------
    function defineModel() {
        let model = {};
        model.vertices = new Float32Array([
            -0.5, -0.5, 0.5,   // 0 - 3 (Front face)
             0.5, -0.5, 0.5,
             0.5,  0.5, 0.5,
            -0.5,  0.5, 0.5,

            -0.5, -0.5, -0.5,   // 4 - 7 (Back face)
             0.5, -0.5, -0.5,
             0.5,  0.5, -0.5,
            -0.5,  0.5, -0.5,
        ]);

        model.vertexColors = new Float32Array([
            0.0, 0.0, 1.0,  // Front face
            0.0, 0.0, 1.0,
            0.0, 0.0, 1.0,
            0.0, 0.0, 1.0,

            1.0, 0.0, 0.0,  // Back face
            1.0, 0.0, 0.0,
            1.0, 0.0, 0.0,
            1.0, 0.0, 0.0,
        ]);

        //
        // CCW winding order
        model.indices = new Uint16Array([
            0, 1, 2, 0, 2, 3,   // Front face
            5, 4, 7, 5, 7, 6,   // Back face
            1, 5, 6, 1, 6, 2,   // Right face
            7, 4, 0, 3, 7, 0,   // Left face
            3, 2, 6, 3, 6, 7,   // Top face
            5, 1, 0, 5, 0, 4    // Bottom face
         ]);

        model.center = {
            x: 0.0,
            y: 0.0,
            z: -2.0
        };

        return model;
    }

    //------------------------------------------------------------------
    //
    // Loads and parses a PLY formatted file into an object ready for
    // rendering.
    //
    //------------------------------------------------------------------
    function load(filename) {
        return new Promise((resolve, reject) => {
            loadFileFromServer(filename)
            .then(fileText => {
                let fileLines = fileText.split('\n');
                let model = defineModel();
                resolve(model);
            })
            .catch(error => {
                reject(error);
            });
        });
    }

    return {
        load : load
    };

}());
