"""Molecular dynamics.

This script should display the atoms (and their connections) in a
molecular dynamics simulation dataset.

You can run the script from the command line by typing
python molecules.py

"""

import vtk
import molecules_io

DATA = "./data/"


# Define a class for the keyboard interface
class KeyboardInterface(object):
    """Keyboard interface.

    Provides a simple keyboard interface for interaction. You may
    extend this interface with keyboard shortcuts for manipulating the
    molecule visualization.

    """

    def __init__(self):
        self.screenshot_counter = 0
        self.render_window = None
        self.window2image_filter = None
        self.png_writer = None
        # Add the extra attributes you need here...

    def keypress(self, obj, event):
        """This function captures keypress events and defines actions for
        keyboard shortcuts."""
        key = obj.GetKeySym()
        if key == "9":
            self.render_window.Render()
            self.window2image_filter.Modified()
            screenshot_filename = ("screenshot%02d.png" %
                                   (self.screenshot_counter))
            self.png_writer.SetFileName(screenshot_filename)
            self.png_writer.Write()
            print("Saved %s" % (screenshot_filename))
            self.screenshot_counter += 1
        # Add your keyboard shortcuts here. If you modify any of the
        # actors or change some other parts or properties of the
        # scene, don't forget to call the render window's Render()
        # function to update the rendering.
        # elif key == ...


# Read data into a vtkPolyData object using the functions in molecules_io.py
data = vtk.vtkPolyData()
data.SetPoints(molecules_io.read_points(DATA + "coordinates.txt"))
data.GetPointData().SetScalars(molecules_io.read_scalars(DATA + "radii.txt"))
data.SetLines(molecules_io.read_connections(DATA + "connections.txt"))

###########
## Spheres
###########

# Source
sphere = vtk.vtkSphereSource()
sphere.SetRadius(0.2)
sphere.SetThetaResolution(8)
sphere.SetPhiResolution(8)

# Glyph
sphere_glyph = vtk.vtkGlyph3D()
sphere_glyph.SetInputData(data)
sphere_glyph.SetSourceConnection(sphere.GetOutputPort())

colorTransferFunction = vtk.vtkColorTransferFunction()
radii = [0.37, 0.68, 0.73, 0.74, 0.77, 2.00]
colorTransferFunction.AddRGBPoint(radii[0], 1.0, 0.0, 0.0)
colorTransferFunction.AddRGBPoint(radii[1], 0.0, 1.0, 0.0)
colorTransferFunction.AddRGBPoint(radii[2], 1.0, 1.0, 0.0)
colorTransferFunction.AddRGBPoint(radii[3], 0.0, 0.0, 1.0)
colorTransferFunction.AddRGBPoint(radii[4], 1.0, 0.0, 1.0)
colorTransferFunction.AddRGBPoint(radii[5], 0.0, 1.0, 1.0)

# Mapper
sphere_mapper = vtk.vtkPolyDataMapper()
sphere_mapper.SetInputConnection(sphere_glyph.GetOutputPort())
sphere_mapper.SetLookupTable(colorTransferFunction)

# Actor
sphere_actor = vtk.vtkActor()
sphere_actor.SetMapper(sphere_mapper)
# sphere_actor.GetProperty().SetColor(0.1,0.0,0.0)

#########
## Tubes
#########

tube_filter = vtk.vtkTubeFilter()
tube_filter.SetInputData(data)
tube_filter.SetRadius(0.05)
tube_filter.SetNumberOfSides(8)
tube_filter.Update()

tube_mapper = vtk.vtkPolyDataMapper()
tube_mapper.SetInputConnection(tube_filter.GetOutputPort())
tube_mapper.Update()
tube_mapper.ScalarVisibilityOff()

tube_actor = vtk.vtkActor()
tube_actor.SetMapper(tube_mapper)
# tube_actor.GetProperty().SetColor(1,0,0)  
# tube_actor.GetProperty().SetOpacity(0.5)

########
# Lines
########

line_mapper = vtk.vtkPolyDataMapper()
line_mapper.SetInputData(data)

line_actor = vtk.vtkActor()
line_actor.SetMapper(line_mapper)
line_actor.GetProperty().SetColor(0.1, 0.0, 0.0)

#########
# Legend
#########

legend = vtk.vtkLegendBoxActor()
legend.SetNumberOfEntries(6)
legend.UseBackgroundOn()
for i in range(6):
    legend.SetEntry( i,
        sphere.GetOutput(),
        f"{radii[i]}",
        colorTransferFunction.GetColor(radii[i])
    )

############
# Rendering
############
renderer = vtk.vtkRenderer()
renderer.SetBackground(0.3, 0.3, 0.3)

#########
# Actors
#########
renderer.AddActor(sphere_actor)
# renderer.AddActor(line_actor)
renderer.AddActor(tube_actor)
renderer.AddActor(legend)

# Create a render window
render_window = vtk.vtkRenderWindow()
render_window.SetWindowName("Molecular dynamics")
render_window.SetSize(500, 500)
render_window.AddRenderer(renderer)

# Create an interactor
interactor = vtk.vtkRenderWindowInteractor()
interactor.SetRenderWindow(render_window)

# Create a window-to-image filter and a PNG writer that can be used
# to take screenshots
window2image_filter = vtk.vtkWindowToImageFilter()
window2image_filter.SetInput(render_window)
png_writer = vtk.vtkPNGWriter()
png_writer.SetInputConnection(window2image_filter.GetOutputPort())

# Set up the keyboard interface
keyboard_interface = KeyboardInterface()
keyboard_interface.render_window = render_window
keyboard_interface.window2image_filter = window2image_filter
keyboard_interface.png_writer = png_writer

# Connect the keyboard interface to the interactor
interactor.AddObserver("KeyPressEvent", keyboard_interface.keypress)

# Initialize the interactor and start the rendering loop
interactor.Initialize()
render_window.Render()
interactor.Start()
