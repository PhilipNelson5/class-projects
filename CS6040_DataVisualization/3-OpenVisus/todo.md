# Part 1

- [X] 1 (CODING) In your Jupyter notebook provide a cell that loads the dataset at resolution 17, timestep 0 and creates a static visualization using matplotlib. [5 pts]
- [X] 2 (CODING) Provide a new cell that loads the dataset at multiple resolutions, timestep 0 and creates an interactive visualization using matplotlib. Your interface should include a slider to change resolution with min resolution = 5, max resolution = 21, and step size = 4. [5 pts]
- [X] 3 (CODING) Provide a new cell that loads the dataset at multiple resolutions, timesteps and creates an interactive visualization using matplotlib. Your interface should include a slider to change resolution and a slider to change time. [5 pts]
- [X] 4 (CODING) In a new cell, extend the interface you created in 3. by adding a threshold slider and modify the visualization to display the image after thresholding (Note: you need to perform thresholding on a grayscale image). Choose appropriate values for min, max and step size for the threshold slider. [5 pts]
- [X] 5 (CODING) Set the threshold value to 210 and identify the pixels above this threshold value as snow. [5 pts]
  - [X] 5.1 Provide a cell that plots the density of snow as a function of time, at resolutions [5, 9, 13, 17, 21]. Comment your observations briefly.
  - [X] 5.2 Provide a cell that computes and plots the error in the density of snow for successive resolutions (i.e. error between resolution 5 and 9, 9 and 13 and so on). Comment your observations briefly. Report the resolutions at which the standard deviation of the error is less than 0.002.
- [X] 6 (CODING) For each resolution you reported in 5.2, compute the range of threshold values for which the density of snow in the Northern Hemisphere is approximately 5-15% during the winter season (December through March). For each resolution, report the mean threshold value. [5 pts]
- [X] 7 (CODING) Provide a cell that interactively plots the density of snow as a function of time. Your interface should include a slider to change the resolution. Note: You should use only the resolutions you found in 5.2 and the corresponding mean threshold values you reported in 6. to compute the density of snow. For each resolution, report the month that received the highest amount of snow and the month that received the lowest amount of snow. [5 pts]
- [X] 8 (CODING) Divide the domain of the dataset into quadrants. Denote the quadrant in top left as Q1 and the quadrant in top right as Q2 (see image below). For resolution 21, with help of an appropriate visualization, find the months in which the density of snow in Q2 is greater than that in Q1. Use the mean threshold value you reported in 6. for resolution 21. [5 pts]

- [X] Question 1: What is the embedding dimension of this dataset? [3 pts]

- [X] Question 2: What is the embedding dimension of the visualization you made in coding exercise 3 and 4? [3 pts]

- [X] Question 3: What is the size of the numpy array in bytes for loading the dataset at resolutions [5, 9, 13, 17, 21], including in memory only timestep 9? [3 pts]

- [X] Question 4: At what resolution can you roughly identify continent boundaries? What happens to the boundaries above and below this resolution? What do you infer from this? [3 pts]

- [X] Question 5: What is the size of the numpy array in bytes for loading the dataset at resolutions [5, 9, 13, 17, 21], including in memory all the timesteps? [3 pts]

- [X] Question 6: Briefly summarize your findings. What did you learn about the science? What did you learn in terms of using the tools while developing the solution of this homework? [5 pts]

# Part 2

- [X] 1 (CODING) In your Jupyter notebook provide a cell that loads the dataset at resolution 24, timestep 15. Save the data to a binary file (using “.raw” extension). [5 pts]

- [X] 2 (CODING) In your Jupyter notebook provide a cell that loads the dataset at multiple resolutions [15, 18, 21, 24], timesteps and creates an interactive visualization using itkwidgets or pyvista. Your interface should include two sliders,
  - [X] (i) a slider to change resolution
  - [X] (ii) a slider to change time [Hint: Refer to itkwidgets example on how to interactively update the viewer]. [5 pts]

- [X] 3 (CODING) Devise an algorithm to estimate percentage thickness of the mixing layer as a function of time for resolutions [15, 18, 21, 24] using z-slices (i.e. plane orthogonal to the z-axis). Provide visualization evidence to support your result and briefly explain your findings. [10 pts] 

- [X] 4 (CODING) Devise an algorithm to estimate percentage thickness of the mixing layer as a function of time for resolutions [15, 18, 21, 24] using
  - [X] (i) x-orthogonal plane in the middle of the volume, as a surrogate to the entire volume 
  - [X] (ii) y-orthogonal plane in the middle of the volume, as a surrogate to the entire volume.
  - [X] Provide visualization evidence to support your result and briefly explain your findings. [10 pts]
  
- [X] 5 (CODING) At resolution 24
  - [X] Plot the magnitude of the two approximations (as functions of time) computed in coding 4 with respect to the computation in coding 3.
  - [X] Compare the runtimes of the algorithm in Coding 3 and Coding 4 and briefly summarize your findings. [10 pts]

- [X] 6 (PARAVIEW) In Paraview
  - [X] open the “.raw” binary file you saved in Coding 1 (using the “Image Reader” importer)
  - [X] visualize it using volume rendering
  - [X] Modify the transfer function to produce a visualization similar to the figure below (note the axis labels are visible)
  - [X] Save the resulting visualization in a png file and the Paraview state file for this view (File-> Save State) in a pvsm file. [10 pts]

- [X] 7 (CODING & PARAVIEW)
  - [X] Build a derived 3d dataset that incorporates the evolution of the mixing layer over time in a single volume
  - [X] visualize it in Paraview
  - [X] (your result should be similar to the figure below).
  - [X] Briefly explain your visualization result and its usefulness. [10 pts]