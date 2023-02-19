from io import TextIOWrapper
import numpy as np


class _Header:
    def __init__(self):
        self.format = ""
        self.n_verts = 0
        self.vert_type = np.single
        self.n_faces = 0

    def __str__(self):
        return f"""Ply Header
format: {self.format}
n vertex: {self.n_verts}
vertex type: {self.vert_type}
n face: {self.n_faces}
"""


def _parse_header(f: TextIOWrapper):
    header = _Header()
    last_element_vertex = False

    while (line := f.readline().strip()) != "end_header":
        tokens = line.split()
        if tokens[0] == "comment":
            continue

        if tokens[0] == "format":
            header.format = "ascii"

        elif tokens[0] == "element":
            n = int(tokens[2])
            if tokens[1] == "vertex":
                header.n_verts = n
                last_element_vertex = True
            elif tokens[1] == "face":
                header.n_faces = n
                last_element_vertex = False
            else:
                raise RuntimeError(f"unknown element {tokens[1]}")

        elif tokens[0] == "property":
            if last_element_vertex:
                if (
                    tokens[2].lower() == "x"
                    or tokens[2].lower() == "y"
                    or tokens[2].lower() == "z"
                ):
                    if tokens[1] == "float":
                        header.vert_type = np.double
                    elif tokens[1] == "double":
                        header.vert_type = np.double
                    else:
                        raise NotImplementedError(
                            f"unimplemented property type {tokens[1]}"
                        )
                else:
                    pass

        else:
            print(f"unimplemented header command {tokens[0]}")

    return header


def _parse_ascii(f: TextIOWrapper, header: _Header):
    verts = np.empty(shape=(header.n_verts, 4), dtype=header.vert_type)
    faces = np.empty(shape=(header.n_faces, 3), dtype=np.intc)

    for i in range(header.n_verts):
        tokens = f.readline().strip().split()
        for j in range(3):
            verts[i, j] = header.vert_type(tokens[j])
        verts[i, 3] = header.vert_type(1)

    for i in range(header.n_faces):
        tokens = f.readline().strip().split()
        n = int(tokens[0])
        if n != 3:
            raise RuntimeError(f"faces must have 3 vertices, found: {n}")
        for j in range(n):
            faces[i, j] = np.intc(tokens[j + 1])

    return (verts, faces)


# https://en.wikipedia.org/wiki/PLY_(file_format)
def read(filename: str):
    """read a .ply file as described in the link above
    This reader does not implement the entire ply standard

    Args:
        filename (str): path to the .ply file
    """
    with open(filename, "r") as f:
        s = f.readline().strip()
        if s != "ply":
            raise RuntimeError("File does not start with 'ply'")

        header = _parse_header(f)

        if header.format == "ascii":
            (verts, faces) = _parse_ascii(f, header)
        else:
            raise NotImplementedError(f"file mode: {header.format} not implemented")

        return verts, faces


if __name__ == "__main__":
    verts, faces = read("./data/bun_zipper_res4.ply")
    print(len(verts), "vertices")
    print(len(faces), "faces")
