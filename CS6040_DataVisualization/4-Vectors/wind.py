"""Air currents.

This script should display a visualization of a vtkStructuredPoints
dataset containing the direction and speed of air currents over North
America.

You can run the script from the command line by typing
python wind.py

"""
import vtk
data_path = './data/'
colors = vtk.vtkNamedColors()

# Define a class for the keyboard interface
class KeyboardInterface(object):
    """Keyboard interface.

    Provides a simple keyboard interface for interaction. You may
    extend this interface with keyboard shortcuts for, e.g., moving
    the slice plane(s) or manipulating the streamline seedpoints.

    Use the arrow keys to move seed point 1
    Hold shift with the arrow keys to move seed point 2
    Press space to cycle through Forward, Backward, and both streamline integration directions
    Use "m" and "l" to add more or less streamlines

    """

    def __init__(self):
        self.screenshot_counter = 0
        self.render_window = None
        self.window2image_filter = None
        self.png_writer = None
        # Add the extra attributes you need here...
        self.point1_x = 19
        self.point1_y = 71
        self.point1_z = 8
        self.point2_x = 59
        self.point2_y = 71
        self.point2_z = 8
        self.streams = None
        self.seeds = None
        self.seed_resolution = 20
        self.shift_r = False
        self.shift_l = False
        self.stream_dir = 0

    def keyrelease(self, obj, event):
        key = obj.GetKeySym()
        if key == 'Shift_R':
            self.shift_r = False
        if key == 'Shift_L':
            self.shift_l = False
        pass

    def keypress(self, obj, event):
        """This function captures keypress events and defines actions for
        keyboard shortcuts."""
        key = obj.GetKeySym()
        # print(key)

        if key == 'Shift_R':
            self.shift_r = True
        elif key == 'Shift_L':
            self.shift_l = True

        elif key == "9":
            self.render_window.Render()
            self.window2image_filter.Modified()
            screenshot_filename = ("screenshot%02d.png" %
                                   (self.screenshot_counter))
            self.png_writer.SetFileName(screenshot_filename)
            self.png_writer.Write()
            print("Saved %s" % (screenshot_filename))
            self.screenshot_counter += 1

        elif key in ["Up", "Down", "Left", "Right"]:
            d = 3
            if key == "Up" and not (self.shift_l or self.shift_r):
                self.point1_x = self.point1_x + d
            elif key == "Up":
                self.point2_x = self.point2_x + d

            elif key == "Down" and not (self.shift_l or self.shift_r):
                self.point1_x = self.point1_x - d
            elif key == "Down":
                self.point2_x = self.point2_x - d

            elif key == "Left" and not (self.shift_l or self.shift_r):
                self.point1_y = self.point1_y + d
            elif key == "Left":
                self.point2_y = self.point2_y + d

            elif key == "Right" and not (self.shift_l or self.shift_r):
                self.point1_y = self.point1_y - d
            elif key == "Right":
                self.point2_y = self.point2_y - d

            self.seeds.SetPoint1(self.point1_x, self.point1_y, self.point1_z)
            self.seeds.SetPoint2(self.point2_x, self.point2_y, self.point2_z)
            self.streams.Update()
            self.render_window.Render()
        
        elif key == "space":
            self.stream_dir = self.stream_dir + 1
            self.stream_dir = self.stream_dir % 3
            if self.stream_dir == 0:
                self.streams.SetIntegrationDirectionToBoth()
            elif self.stream_dir == 1:
                self.streams.SetIntegrationDirectionToForward()
            else:
                self.streams.SetIntegrationDirectionToBackward()
            
            self.streams.Update()
            self.render_window.Render()

        elif key == "m":
            self.seed_resolution = self.seed_resolution + 5
            self.seed_resolution = max(0, self.seed_resolution)
            self.seeds.SetResolution(self.seed_resolution)
            self.streams.Update()
            self.render_window.Render()
        elif key == "l":
            self.seed_resolution = self.seed_resolution - 5
            self.seed_resolution = min(100, self.seed_resolution)
            self.seeds.SetResolution(self.seed_resolution)
            self.streams.Update()
            self.render_window.Render()


def createImagePlaneWidget(reader, picker, renderer, interactor, axis, slice_idx):
    a, b = reader.GetOutput().GetScalarRange()
    
    ctf = vtk.vtkColorTransferFunction()
    ctf.AddRGBPoint(a, 0, 0, 0)
    ctf.AddRGBPoint(b, 1, 0, 0)

    ipw = vtk.vtkImagePlaneWidget()
    ipw.SetPicker(picker)
    ipw.SetInputData(reader.GetOutput())
    ipw.SetCurrentRenderer(renderer)
    ipw.SetInteractor(interactor)

    ipw.PlaceWidget()
    if axis == 'x':
        ipw.SetPlaneOrientationToXAxes()
    if axis == 'y':
        ipw.SetPlaneOrientationToYAxes()
    if axis == 'z':
        ipw.SetPlaneOrientationToZAxes()
    ipw.SetSliceIndex(slice_idx)
    ipw.DisplayTextOn()
    ipw.EnabledOn()
    return ipw


# Read the dataset
reader = vtk.vtkStructuredPointsReader()
reader.SetFileName(data_path + "wind.vtk")
reader.Update()
W,H,D = reader.GetOutput().GetDimensions()

#
#
# Add your code here...
#
#
# Renderer
###############################################################################
renderer = vtk.vtkRenderer()
renderer.SetBackground(0.2, 0.2, 0.2)

# Create a render window
render_window = vtk.vtkRenderWindow()
render_window.SetWindowName("Air currents")
render_window.SetSize(1800, 1600)
render_window.AddRenderer(renderer)

# Create an interactor
interactor = vtk.vtkRenderWindowInteractor()
interactor.SetRenderWindow(render_window)

# Create a picker
picker = vtk.vtkCellPicker() # use same picker for all
picker.SetTolerance(0.005)

# outline
###############################################################################
outline = vtk.vtkOutlineFilter()
outline.SetInputData(reader.GetOutput())
outline_mapper = vtk.vtkPolyDataMapper()
outline_mapper.SetInputConnection(outline.GetOutputPort())
outline_actor = vtk.vtkActor()
outline_actor.SetMapper(outline_mapper)
outline_actor.GetProperty().SetColor(colors.GetColor3d('White'))

# slice
###############################################################################
slice_mapper = vtk.vtkImageSliceMapper()
slice_mapper.SetInputConnection(reader.GetOutputPort())
slice_mapper.SliceAtFocalPointOn()
image_slice = vtk.vtkImageSlice()
image_slice.SetMapper(slice_mapper)

# arrows
###############################################################################
plane = vtk.vtkPlane()
plane.SetOrigin(W//2,H//2,D//2+1)

# plane.SetNormal(1,0,0)
# plane.SetNormal(0,1,0)
plane.SetNormal(0,0,1)

#create cutter
cutter=vtk.vtkCutter()
cutter.SetCutFunction(plane)
cutter.SetInputConnection(reader.GetOutputPort())
cutter.Update()

arrow = vtk.vtkArrowSource()
glyphs = vtk.vtkGlyph3D()
glyphs.SetInputData(cutter.GetOutput())
glyphs.SetSourceConnection(arrow.GetOutputPort())
# glyphs.SetInputConnection(threshold.GetOutputPort())

glyphs.SetVectorModeToUseVector()
glyphs.SetScaleModeToDataScalingOff()
glyphs.SetScaleFactor(1)
glyphs.SetColorModeToColorByVector()

glyph_mapper =  vtk.vtkPolyDataMapper()
glyph_mapper.SetInputConnection(glyphs.GetOutputPort())
glyph_actor = vtk.vtkActor()
glyph_actor.SetMapper(glyph_mapper)

glyph_mapper.UseLookupTableScalarRangeOn()

glyphs.Update()
s0,sf = glyphs.GetOutput().GetScalarRange()
lut = vtk.vtkColorTransferFunction()
lut.AddRGBPoint(s0, 0,0,1)
lut.AddRGBPoint(1 * (s0+sf)/5, 0,1,1)
lut.AddRGBPoint(2 * (s0+sf)/5, 0,1,0)
lut.AddRGBPoint(3 * (s0+sf)/5, 1,1,0)
lut.AddRGBPoint(sf, 1,0,0)
glyph_mapper.SetLookupTable(lut)

# Scalar Bar
###############################################################################
scalar_bar = vtk.vtkScalarBarActor()
scalar_bar.SetOrientationToHorizontal()
scalar_bar.SetLookupTable(lut)
scalar_bar.SetTitle('Wind Speed')

# create the scalar_bar_widget
scalar_bar_widget = vtk.vtkScalarBarWidget()
scalar_bar_widget.SetInteractor(interactor)
scalar_bar_widget.SetScalarBarActor(scalar_bar)
scalar_bar_widget.On()

# Stream tracer
###############################################################################
x = 18
y = 71
z = D//2+1

seeds = vtk.vtkLineSource()
seeds.SetPoint1(x+1, y, z)
seeds.SetPoint2(x+41, y, z)
seeds.SetResolution(20)

seed_outline_mapper = vtk.vtkPolyDataMapper()
seed_outline_mapper.SetInputConnection(seeds.GetOutputPort())
seed_outline_actor = vtk.vtkActor()
seed_outline_actor.SetMapper(seed_outline_mapper)
seed_outline_actor.GetProperty().SetColor(colors.GetColor3d('Red'))

stream_tracer = vtk.vtkStreamTracer()
stream_tracer.SetInputData(reader.GetOutput())
stream_tracer.SetSourceConnection(seeds.GetOutputPort())
stream_tracer.SetMaximumPropagation(1000)
stream_tracer.SetMaximumNumberOfSteps(5000)
stream_tracer.SetInitialIntegrationStep(0.1)
stream_tracer.SetIntegrationDirectionToBoth()

stream_tube = vtk.vtkTubeFilter()
stream_tube.SetInputConnection(stream_tracer.GetOutputPort())
stream_tube.SetRadius(.3)
stream_tube.SetNumberOfSides(5)

streamline_mapper = vtk.vtkPolyDataMapper()
streamline_mapper.SetInputConnection(stream_tube.GetOutputPort())
streamline_mapper.SetLookupTable(lut)
streamline_actor = vtk.vtkActor()
streamline_actor.SetMapper(streamline_mapper)
streamline_actor.VisibilityOn()



# Add Actors to the Renderer
###############################################################################
renderer.AddActor(outline_actor)
renderer.AddActor(seed_outline_actor)
# renderer.AddActor(image_actor)
renderer.AddViewProp(image_slice)
renderer.AddActor(glyph_actor)
renderer.AddActor(streamline_actor)


# ipw = createImagePlaneWidget(reader, picker, renderer, interactor, 'z', D//2)

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
keyboard_interface.seeds = seeds
keyboard_interface.streams = stream_tracer

# Connect the keyboard interface to the interactor
interactor.AddObserver("KeyPressEvent", keyboard_interface.keypress)
interactor.AddObserver("KeyReleaseEvent", keyboard_interface.keyrelease)

# Initialize the interactor and start the rendering loop
interactor.Initialize()
render_window.Render()
renderer.GetActiveCamera().SetViewUp(1.0, 0.0, 0)
interactor.Start()
