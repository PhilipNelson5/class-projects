double interpalate(int i, double n, double min, double max) {
	return (i * (max - min) / n) + min;
}

int mandelbrot(int i, int j, int IMAGE_WIDTH, int IMAGE_HIGHT, int MAX_ITERS,
	double X_MIN, double X_MAX, double Y_MIN, double Y_MAX) {

	double xtemp;
	double x0 = interpalate(j, IMAGE_WIDTH, X_MIN, X_MAX);
	double y0 = interpalate(i, IMAGE_HIGHT, Y_MIN, Y_MAX);
	double x = 0.0;
	double y = 0.0;
	int iters = 0;
	while (x*x + y*y < 4 && iters < MAX_ITERS) {
		xtemp = x*x - y*y + x0;
		y = 2 * x*y + y0;
		x = xtemp;
		iters += 1;
	}
	return iters;
}
