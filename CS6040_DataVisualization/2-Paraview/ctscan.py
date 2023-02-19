# File:        ctscan.py
# Description: MPR rendering

import vtk
DATA = "./data/"

colors = vtk.vtkNamedColors()


# image reader
filename1 = DATA + "ctscan_ez.vtk"
reader1 = vtk.vtkStructuredPointsReader()
reader1.SetFileName( filename1 )
reader1.Update()

W,H,D = reader1.GetOutput().GetDimensions()
a1,b1 = reader1.GetOutput().GetScalarRange()
print("Range of image: %d--%d" %(a1,b1))

filename2 = DATA + "ctscan_ez_bin.vtk"
reader2 = vtk.vtkStructuredPointsReader()
reader2.SetFileName( filename2 )
reader2.Update()

a2,b2 = reader2.GetOutput().GetScalarRange()
print("Range of segmented image: %d--%d" %(a2,b2))

# renderer and render window
ren = vtk.vtkRenderer()
ren.SetBackground(.2, .2, .2)
renWin = vtk.vtkRenderWindow()
renWin.SetSize( 512, 512 )
renWin.AddRenderer( ren )

# render window interactor
iren = vtk.vtkRenderWindowInteractor()
iren.SetRenderWindow( renWin )

#
# add your code here for MPR and the liver surface
#
# Help to get started...
#
picker=vtk.vtkCellPicker() # use same picker for all
picker.SetTolerance(0.005)

def createImagePlaneWidget(axis, idx):
    ipw = vtk.vtkImagePlaneWidget()
    ipw.SetPicker(picker)
    ipw.SetInputData(reader1.GetOutput())
    ipw.SetCurrentRenderer(ren)
    ipw.SetInteractor(iren)
    ipw.PlaceWidget()
    if axis == 'x':
        ipw.SetPlaneOrientationToXAxes()
    if axis == 'y':
        ipw.SetPlaneOrientationToYAxes()
    if axis == 'z':
        ipw.SetPlaneOrientationToZAxes()
    ipw.SetSliceIndex(idx)
    ipw.DisplayTextOn()
    ipw.EnabledOn()
    return ipw

ipwx = createImagePlaneWidget('x', int(W/2))
ipwy = createImagePlaneWidget('y', int(H/2))
ipwz = createImagePlaneWidget('z', int(D/2))

### INSERT YOUR CODE HERE
iso = vtk.vtkContourFilter()
iso.SetInputConnection(reader2.GetOutputPort())
iso.SetValue(0, 255)

isoMapper = vtk.vtkPolyDataMapper()
isoMapper.SetInputConnection(iso.GetOutputPort())
isoMapper.ScalarVisibilityOff()

isoActor = vtk.vtkActor()
isoActor.SetMapper(isoMapper)
isoActor.GetProperty().SetColor(colors.GetColor3d("Orange"))

###

# create an outline of the dataset
outline = vtk.vtkOutlineFilter()
outline.SetInputData( reader1.GetOutput() )
outlineMapper = vtk.vtkPolyDataMapper()
outlineMapper.SetInputData( outline.GetOutput() )
outlineActor = vtk.vtkActor()
outlineActor.SetMapper( outlineMapper )

# the actors property defines color, shading, line width,...
outlineActor.GetProperty().SetColor(0.8,0.8,0.8)
outlineActor.GetProperty().SetLineWidth(2.0)

# add the actors
ren.AddActor( outlineActor )
ren.AddActor( isoActor )
## ADD YOUR ACTORS HERE

##
renWin.Render()

# create window to image filter to get the window to an image
w2if = vtk.vtkWindowToImageFilter()
w2if.SetInput(renWin)

# create png writer
wr = vtk.vtkPNGWriter()
wr.SetInputData(w2if.GetOutput())

# Python function for the keyboard interface
count = 0
liver_visibility = True
def Keypress(obj, event):
    global count, liver_visibility
    key = obj.GetKeySym()
    if key == "s":
        renWin.Render()
        w2if.Modified() # tell the w2if that it should update
        fnm = "screenshot%02d.png" %(count)
        wr.SetFileName(fnm)
        wr.Write()
        print("Saved '%s'" %(fnm))
        count = count+1
    elif key == 'l':
        liver_visibility = not liver_visibility
        isoActor.SetVisibility(liver_visibility)
        renWin.Render()

# add keyboard interface, initialize, and start the interactor
iren.AddObserver("KeyPressEvent", Keypress)
iren.Initialize()
iren.Start()
