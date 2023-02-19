import struct
import numpy as np

# https://web.cs.ucdavis.edu/~okreylos/PhDStudies/Spring2000/ECS277/DataSets.html

def read(filename: str) -> np.ndarray:
    """read a .vol file as described in the link above

    Args:
        filename (str): path to the .vol file
    """
    def readn(f, n, endian, type_str): 
        return struct.unpack(endian + (type_str * n), f.read(n*4))

    with open(filename, "rb") as f:
        shape = readn(f, 3, '>', 'i')
        _ = readn(f, 1, '>', 'i')
        extent = readn(f, 3, '>', 'f')
        data = np.fromfile(f, dtype=">b")
        data = data.reshape(shape)
        
        return data