# [X] Part 1: Getting familiar with Matplotlib

## [X] A
- [X] 1. Create an array with 200 elements from 1 to 200 in order.
- [X] 2. Create a line plot to visualize this array of 200 elements

## [X] B
- [X] 1. Create an array x with 10,000 floats in the range [1,10].
- [X] 2. Plot, for the array x, a histogram showing a uniform distribution of bin sizes (similar to the figure shown below) (optional: use “rwidth” in plt.hist to create space between bins, see matplotlib documentation for details).
- [X] 3. For the same array x, plot a histogram showing a monotonically increasing distribution of bin sizes.
- [X] 4. For the same array x, plot a histogram showing a monotonically decreasing distribution of bin sizes.
- [X] 5. Which of the histograms you created before can be considered a misleading visualizations, and why?

## [X] C
- [X] 1. Create an array by generating 100 random numbers with normal (aka Gaussian) distribution.
- [X] 2. Write the numbers out to a binary file (using numpy). Read the binary file back into an array (using numpy).
- [X] 3. Create a histogram and a bar chart of the data you read back from the binary file.
- [X] 4. What information does the histogram shows that the bar chart does not?
- [X] 5. What information does the bar chart shows that the histogram does not?
 
# [X] Part 2: Interesting datasets for visualization

## [X] A Download the NOAA Land Ocean Temperature Anomalies Data Set. Load the data.
- [X] 1. Create a Scatter Plot and a Bar Plot. Include a label called “Year” along the x-axis and a label called “Degrees Celsius +/- From Average” along the y-axis.
- [X] 2. Describe the trends you see in the data.
- [X] 3. Discuss which plot you believe shows those trends better and why.
- [X] 4. Provide an example of a different plot which better represents the long-term trends.

## [X] B Download the statistical data about marriage from: https://raw.githubusercontent.com/fivethirtyeight/data/master/marriage/both_sexes.csv. Read carefully the description of the dataset on the webpage. Load the data.
- [X] 1. Create a Star Plot (polar plot) and a Line Graph using at least three fields.
- [X] 2. Describe the trends you see in the plots for the three different fields.
- [X] 3. What are some pros and cons of using a Star Plot vs a Line Graph?

## [X] C Download the U.S. Birth data set: https://raw.githubusercontent.com/fivethirtyeight/data/master/births/US_births_2000-2014_SSA.csv. Load the data.
Create visualizations to support your answers for the following questions:
- [X] 1. What month had the highest number of births?
- [X] 2. What month had the lowest number of births?
- [X] 3. Are there any interesting trends in the data?

## [X] D. Five Thirty Eight maintains a server with many interesting datasets: https://github.com/fivethirtyeight/data (Links to an external site.). Choose 1 data set and:
- [W] 1. Produce three good visualizations which convey a unique trend in the data. Discuss the trends you see briefly.
- [X] 2. Produce three bad visualizations and explain briefly why these visualizations are not suitable for the dataset you picked.

# [ ] Part 3: Introduction to 3D scalar field datasets

## [ ] A. Python/Jupyter also can be used for analysis and visualization of 3D scalar field datasets, such as the brain MRI images.

- [X] 1. Download the brain MRI dataset “T2.raw  Download T2.raw.zip” from canvas, unzip it and load the file "T2.raw" into a numpy array. The data format is float32 with dimensions 320 x 320 x 256.
- [X] 2. Extract one slice from the volume and save it as a PNG image.

## [X] B. Create a ipywidget slider to threshold the data for a given slice. For increasing threshold values, your output should look similar to the visualizations below (i.e., the figure shows only values in the data below the selected threshold value):

## [X] C.

- [X] 1. What is the topological dimensionality of this dataset?
- [X] 2. What is the geometrical dimensionality of the visualization in B (the one in the picture above)?

# [X] Part 4: Read The Value of Visualization Paper

## [X] A.

- [X] 1. Why is assessing value of visualizations important?
- [X] 2. What are the two measures for deciding the value of visualizations?

## [X] B. Briefly describe the mathematical model for the visualization block shown in Fig.1.

## [X] C. State four parameters that describe the costs associated with any visualization technique.

## [X] D. What are the pros and cons of interactivity of visualizations?