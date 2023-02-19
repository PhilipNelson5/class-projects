What to turn in: Provide a short report answering any specific question asked below. Be sure to explain any figures you submit and refer to them in the answers. Remember, this is a visualization class and the answers should be supported by data analysis and visualization evidence. Submit your report, ParaView state files and python scripts to Canvas in a single zip file (provide your first_lastname in the filename).

# Part 1: Scalar Field Visualization

## Question 1. [15 pts] Explore the terrain model of Honolulu, Hawaii. For each item below, you must include a screenshot of the ParaView render view in your report. 

### A.
Load honolulu.vtk in ParaView.
- [X] What is the datatype ParaView assigns when loading the data?
- [X] In which ParaView panel can you find the data type?
- [X] Briefly explain advantages and disadvantages of using this data type compared to possible alternatives. [5 pts]

### B.
- [X] Find and apply an appropriate filter to color the dataset using height values. The result should look like the picture below. [5 pts]

### C.
- [X] Use a contour filter represented as points of size 5.
- [X] Find and report the isovalue that contains the three highest peaks in the dataset. [5 pts]

- [X] Save and submit the ParaView state file for this question in question1.pvsm. 

## Question 2. [20 pts]
In this question, you will explore a 3d CT Image of a human skull. For each item below, you must include a screenshot of the ParaView render view in your report.

### A.
Load headsq.vti in ParaView.
- [X] What is the data type ParaView assigns when loading the data? [5 pts]

### B.
- [X] Split the render view into two columns.
- [X] Display the histogram of the scalar field in one of the columns and be careful to choose an appropriate bin size.
- [X] In the other column, visualize the dataset using a volume rendering.
- [X] Modify the color map to highlight Skin, Skull and Teeth (use the histogram to guide you through this process). [5 pts]

### C.
- [X] Hide the histogram view and split the render view by columns again.
- [X] Display the isosurfaces that correspond to Skin, Skull and Teeth in the new view.
- [X] Use one contour filter per isovalue and set an appropriate opacity value in each contour filter to visualize all the isosurfaces simultaneously.
- [X] Explain the criterion you used to choose the isovalues. [5 pts]

### D.
- [X] Split the view with volume rendering into two rows.
- [X] Display three orthogonal cross sections of the dataset in the middle of the volume in the new view and rename this view as “Slice view”.
- [X] Rotate the dataset and look at the slices from different viewpoints.
- [X] Finally, link the camera of the “Slice view” with the isosurface and the volume rendering view. [5 pts]

- [X] Save and submit the ParaView state file for this question in question2.pvsm.

# Part 2: Vector Field Visualization

[15 pts] In this question, you will visualize vector fields of WRF forecast simulations of Hurricane Katrina’s path. The dataset Hurricane_Katrina.vts has 4 fields (T, QCLOUD, QVAPOR, and Wind).
T is the Temperature,
QCLOUD is the cloud water mixing ratio,
QVAPOR is the Column Water Vapor Content,
and Wind is Wind Speed.
For each item below, you must include a screenshot of the ParaView render view in your report.

## A.
- [X] Load Hurricane_Katrina.vts in ParaView.
- [X] Create streamlines of the wind flow within the hurricane.
- [X] To get a good overview of the flow, you need to seed streamlines at multiple positions.
- [X] Use both point and line source to create a compelling visualization.
- [X] Adjust various parameters of streamlines to reduce clutter. [5 pts]

## B.
- [X] Add cone glyphs to the streamlines to see the direction of the flow and scale them appropriately. [5 pts]

## C.
- [X] To get an overall sense of the direction of the wind flow within the hurricane, create a visualization using arrow glyphs at randomly sampled places throughout the volume.
- [X] You should use enough arrow glyphs to get a good overview, but not so many arrow glyphs that the view is cluttered. [5 pts]

Save and submit the ParaView state file for this question in question3.pvsm.

# Part 3: VTK (note: you have to install in your python environment the vtk package)

## Question 1. [30 pts]

The data from a molecular dynamics simulation (provided by Daniel Spångberg) are stored in three files: coordinates.txt contains 3D coordinates of a number of atoms, radii.txt contains the radii of these atoms, and connections.txt defines how the atoms are connected to each other.

One way to visualize the molecules is to represent each atom with a sphere with a radius corresponding to the radius of the atom. Some of the atoms have quite similar radii, so to make it easier to distinguish between these we can color-map the spheres depending on their radius. Finally, the connections between the atoms can be made with tubes. The following image show an example of what the result can look like (note: this is not necessarily the correct solution).

**You can use this python script to start from: molecules-start.zip Please attach both your modified script and screenshots of your working applications.**

Given those datasets you will:

### A.
- [X] Display the atoms with some geometric primitives as described above.
- [X] The size and color of the atoms should be modulated depending on atom radii. [10 pts]

### B.
- [X] Display atom connections with lines or tubes. [10 pts]

### C.
- [X] Add a legend or a colorbar that shows the radius-to-color mapping. [10 pts]

## Question 2. [30 pts]

Visualize a CT-scan of the abdominal region of a human and a pre-segmented “mask” that shows where in the volume the liver is located. You shall create a small application that can be used to compare the mask with the original volume. Such an application could for example be used by a physician to verify that the mask, which might have been generated with an automatic segmentation algorithm, is correct.

The CT scan is stored as a vtkStructuredPoints dataset of signed 16-bit data (short) representing Hounsfield units. The segmented liver is represented as a binary 8-bit (unsigned char) volume where the liver voxels have the value 255 and the background voxels have the value 0.

**A Python script to start from can be found here ctscan.py Please attach both your modified script and screenshots of your working applications.**

Given those datasets you will:

### A.
- [X] Display the segmented liver mask as a solid surface selecting a proper isovalue. Report a figure of your result and comment on the isovalue that you chose. [10 pts]

### B.
- [X] Display the CT-scan with multi planar reformatting (MPR). Three axes oriented planes are OK. I recommend you to use vtkImagePlaneWidget that provide means to browse and rotate the plane, change contrast, and to probe the data. See example in the script. [10 pts]

### C.
- [X] Probe the image with the mouse cursor and examine the density values of bone, air, and soft tissue. Report the values of those. [5 pts]
- [X] Provide a keyboard interface to switch on/off the rendering of the segmented liver (can be controlled with the actor). [5 pts]

Note: you can look at some VTK python filter examples here: https://kitware.github.io/vtk-examples/site/Python/