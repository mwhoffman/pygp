#include <stdlib.h>
#include <math.h>


/******************************************************************************
 * Definitions of the code to compute the covariance matrix. This is pulled
 * directly from Miguel's code---hence why it stretches beyond 80 characters per
 * line! Ewww! :)
 ******************************************************************************/

/* Function that evaluates cov[x_i,x_j] */

double cov(double *x_i, double *x_j, double sigma, double *l, int d) {

	int i;

	double squaredDistance = 0;
	for (i = 0 ; i < d ; i++)
		squaredDistance += (x_i[ i ] - x_j[ i ]) * (x_i[ i ] - x_j[ i ]) * l[ i ];

	return sigma * exp(-0.5 * squaredDistance);
}

/* Function that evaluates d cov[x_i,x_j] / [ dx_j[k] ] */

double dcovdxjk(double *x_i, double *x_j, double sigma, double *l, int d, int k) {

	return l[ k ] * (x_i[ k ] - x_j[ k ]) * cov(x_i, x_j, sigma, l, d);
}

/* Function that evaluates d cov[x_i,x_j] / [ dx_i[m] ] */

double dcovdxim(double *x_i, double *x_j, double sigma, double *l, int d, int m) {

	return -l[ m ] * (x_i[ m ] - x_j[ m ]) * cov(x_i, x_j, sigma, l, d);
}

/* Function that evaluates d^2 cov[x_i,x_j] / [ dx_j[k] dx_i[m] ] */

double d2covdxjkxim(double *x_i, double *x_j, double sigma, double *l, int d, int k, int m) {

	return (m == k ? l[ m ] : 0) * cov(x_i, x_j, sigma, l, d) +
		l[ k ] * (x_i[ k ] - x_j[ k ]) * dcovdxim(x_i, x_j, sigma, l, d, m);
}

/* Function that evaluates d^2 cov[x_i,x_j] / [ dx_j[k] dx_j[lIndex] ] */

double d2covdxjkxjl(double *x_i, double *x_j, double sigma, double *l, int d, int k, int lIndex) {

	return -(k == lIndex ? l[ k ] : 0) * cov(x_i, x_j, sigma, l, d) +
		l[ k ] * (x_i[ k ] - x_j[ k ]) * dcovdxjk(x_i, x_j, sigma, l, d, lIndex);
}

/* Function that evaluates d^2 cov[x_i,x_j] / [ dx_i[m] dx_i[n] ] */

double d2covdximxin(double *x_i, double *x_j, double sigma, double *l, int d, int m, int n) {

	return -(n == m ? l[ n ] : 0) * cov(x_i, x_j, sigma, l, d) -
		l[ m ] * (x_i[ m ] - x_j[ m ]) * dcovdxim(x_i, x_j, sigma, l, d, n);
}

/* Function that evaluates d^3 cov[x_i,x_j] / [ dx_j[k] dx_i[m] dx_i[n] ] */

double d3covdxjkximxin(double *x_i, double *x_j, double sigma, double *l, int d, int k, int m, int n) {

	return (m == k ? l[ k ] : 0) * dcovdxim(x_i, x_j, sigma, l, d, n) +
		(n == k ? l[ k ] : 0) * dcovdxim(x_i, x_j, sigma, l, d, m) +
		l[ k ] * (x_i[ k ] - x_j[ k ]) * d2covdximxin(x_i, x_j, sigma, l, d, m, n);
}

/* Function that evaluates d^3 cov[x_i,x_j] / [ dx_j[k] dx_j[lIndex] dx_i[m] ] */

double d3covdxjkxjlxim(double *x_i, double *x_j, double sigma, double *l, int d, int k, int lIndex, int m) {

	return -(lIndex == k ? l[ k ] : 0) * dcovdxim(x_i, x_j, sigma, l, d, m) +
		(m == k ? l[ k ] : 0) * dcovdxjk(x_i, x_j, sigma, l, d, lIndex) +
		l[ k ] * (x_i[ k ] - x_j[ k ]) * d2covdxjkxim(x_i, x_j, sigma, l, d, lIndex, m);
}

/* Function that evaluates d^4 cov[x_i,x_j] / [ dx_j[k] dx_j[lIndex] dx_i[m] dx_i[n] ] */

double d4covdxjkxjlximxin(double *x_i, double *x_j, double sigma, double *l, int d, int k, int lIndex, int m, int n) {

	return -(lIndex == k ? l[ k ] : 0) * d2covdximxin(x_i, x_j, sigma, l, d, m, n) +
		(m == k ? l[ k ] : 0) * d2covdxjkxim(x_i, x_j, sigma, l, d, lIndex, n) +
		(n == k ? l[ k ] : 0) * d2covdxjkxim(x_i, x_j, sigma, l, d, lIndex, m) +
		l[ k ] * (x_i[ k ] - x_j[ k ]) * d3covdxjkximxin(x_i, x_j, sigma, l, d, lIndex, m, n);
}

/********** WE COMPUTE THE COVARIANCES BETWEEN THE DATA POINTS AND EVERYTHING ELSE **********/

/* Function which fills in the covariance entries between the n data points */

void computeCovDataPointsDataPoints(double *ret, double *X, int n, int nrow, int d, double *l, double sigma, double sigma0) {

	int i, j, k;

	/* We allocate memory to store the data points */

	double *x1 = (double *) malloc(sizeof(double) * d);
	double *x2 = (double *) malloc(sizeof(double) * d);

	/* We compute the covariance function at the n available observations */

	for (i = 0 ; i <  n ; i++) {
		for (j = i ; j < n ; j++) {

			/* We store the data points at which to evaluate the covariance function */

			for (k = 0 ; k < d ; k++) {
				x1[ k ] = X[ i + n * k ];
				x2[ k ] = X[ j + n * k ];
			}

			/* We evaluate the covariance function */

			ret[ i + j * nrow ] = cov(x1, x2, sigma, l, d);

			if (j == i)
				ret[ i + j * nrow ] = ret[ i + j * nrow] + sigma0;

			ret[ j + i * nrow ] = ret[ i + j * nrow ];
		}
	}

	free(x1); free(x2);
}

/* Function which fills in the covariance entries between the n data points and the gradient at the minimum */

void computeCovDataPointsGradient(double *ret, double *X, int n, int nrow, int d, double *l, double sigma, double *m) {

	int i, j, k;

	/* We allocate memory to store the data points */

	double *x1 = (double *) malloc(sizeof(double) * d);

	/* We evaluate the covariance function at the n available observations and d gradient variables */

	for (i = 0 ; i < n ; i++) {
		for (j = 0 ; j < d ; j++) {

			/* We store the data points at which to evaluate the covariance function */

			for (k = 0 ; k < d ; k++)
				x1[ k ] = X[ i + n * k ];

			/* We evaluate the covariance function */

			ret[ i + (n + j) * nrow ] = dcovdxjk(x1, m, sigma, l, d, j);
			ret[ (n + j) + i * nrow ] = ret[ i + (n + j) * nrow ];
		}
	}

	free(x1);
}

/* Function which fills in the covariance entries between the n data points and the non diagonal hessian elements */

void computeCovDataPointsNonDiagHessian(double *ret, double *X, int n, int nrow, int d, double *l, double sigma, double *m) {

	int i, j, k, h, counter;

	/* We allocate memory to store the data points */

	double *x1 = (double *) malloc(sizeof(double) * d);

	for (i = 0 ; i < n ; i++) {
		counter = 0;
		for (j = 0 ; j < d ; j++) {
			for (h = j + 1 ; h < d ; h++) {

				/* We store the data points at which to evaluate the covariance function */

				for (k = 0 ; k < d ; k++)
					x1[ k ] = X[ i + n * k ];

				/* We evaluate the covariance function */

				ret[ i + (n + d + counter) * nrow ] = d2covdxjkxjl(x1, m, sigma, l, d, j, h);
				ret[ (n + d + counter) + i * nrow ] = ret[ i + (n + d + counter) * nrow ];

				counter++;
			}
		}
	}

	free(x1);
}

/* Function which fills in the covariance entries between the n data points and the diagonal hessian elements */

void computeCovDataPointsDiagHessian(double *ret, double *X, int n, int nrow, int d, double *l, double sigma, double *m) {

	int i, j, k;

	/* We allocate memory to store the data points */

	double *x1 = (double *) malloc(sizeof(double) * d);

	for (i = 0 ; i < n ; i++) {
		for (j = 0 ; j < d ; j++) {

			/* We store the data points at which to evaluate the covariance function */

			for (k = 0 ; k < d ; k++)
				x1[ k ] = X[ i + n * k ];

			/* We evaluate the covariance function */

			ret[ i + (n + d + d * (d - 1) / 2 + j) * nrow ] = d2covdxjkxjl(x1, m, sigma, l, d, j, j);
			ret[ (n + d + d * (d - 1) / 2 + j) + i * nrow ] = ret[ i + (n + d + d * (d - 1) / 2 + j) * nrow ];
		}
	}

	free(x1);
}

/* Function which fills in the covariance entries between the n data points and the minimum */

void computeCovDataPointsMinimum(double *ret, double *X, int n, int nrow, int d, double *l, double sigma, double *m) {

	int i, k;

	/* We allocate memory to store the data points */

	double *x1 = (double *) malloc(sizeof(double) * d);

	for (i = 0 ; i < n ; i++) {

		/* We store the data points at which to evaluate the covariance function */

		for (k = 0 ; k < d ; k++)
			x1[ k ] = X[ i + n * k ];

		/* We evaluate the covariance function */

		ret[ i + (n + d + d * (d - 1) / 2 + d) * nrow ] = cov(x1, m, sigma, l, d);
		ret[ (n + d + d * (d - 1) / 2 + d) + i * nrow ] = ret[ i + (n + d + d * (d - 1) / 2 + d) * nrow ];
	}

	free(x1);
}

/********** WE COMPUTE THE COVARIANCES BETWEEN THE GRADIENT AND EVERYTHING ELSE **********/

/* Function which fills in the covariance entries between the gradient entries and the gradient entries */

void computeCovGradientGradient(double *ret, int n, int nrow, int d, double *l, double sigma, double *m) {

	int i, j;

	for (i = 0 ; i < d ; i++) {
		for (j = i ; j < d ; j++) {
			ret[ (n + i) + (n + j) * nrow ] = d2covdxjkxim(m, m, sigma, l, d, j, i);
			ret[ (n + j) + (n + i) * nrow ] = ret[ (n + i) + (n + j) * nrow ];
		}
	}
}


/* Function which fills in the covariance entries between the gradient entries and the non diagonal entries of the hessian */

void computeCovGradientNonDiagHessian(double *ret, int n, int nrow, int d, double *l, double sigma, double *m) {

	int i, j, h, counter;

	for (i = 0 ; i < d ; i++) {
		counter = 0;
		for (j = 0 ; j < d ; j++) {
			for (h = j + 1 ; h < d ; h++) {

				ret[ (n + i) + (n + d + counter) * nrow ] =
					d3covdxjkxjlxim(m, m, sigma, l, d, j, h, i);
				ret[ (n + d + counter) + (n + i ) * nrow ] = ret[ (n + i) + (n + d + counter) * nrow ];

				counter++;
			}
		}
	}
}

/* Function which fills in the covariance entries between the gradient entries and the diagonal entries of the hessian */

void computeCovGradientDiagHessian(double *ret, int n, int nrow, int d, double *l, double sigma, double *m) {

	int i, j;

	for (i = 0 ; i < d ; i++) {
		for (j = 0 ; j < d ; j++) {
			ret[ (n + i) + (n + d + d * (d - 1) / 2 + j) * nrow ] = d3covdxjkxjlxim(m, m, sigma, l, d, j, j, i);
			ret[ (n + d + d * (d - 1) / 2 + j) + (n + i) * nrow ] = ret[ (n + i) + (n + d + d * (d - 1) / 2 + j) * nrow ];
		}
	}
}

/* Function which fills in the covariance entries between the gradient entries and the minimum */

void computeCovGradientMinimum(double *ret, int n, int nrow, int d, double *l, double sigma, double *m) {

	int i;

	for (i = 0 ; i < d ; i++) {
		ret[ (n + i) + (n + d + d * (d - 1) / 2 + d) * nrow ] = dcovdxjk(m, m, sigma, l, d, i);
		ret[ (n + d + d * (d - 1) / 2 + d) + (n + i) * nrow ] = ret[ (n + i) + (n + d + d * (d - 1) / 2 + d) * nrow ];
	}
}

/********** WE COMPUTE THE COVARIANCES BETWEEN THE NON DIAGONAL HESSIAN AND EVERYTHING ELSE **********/

/* Function which fills in the covariance entries between the non diagonal hesssian entries and the non diagonal hessian entries */

void computeCovNonDiagHessianNonDiagHessian(double *ret, int n, int nrow, int d, double *l, double sigma, double *m) {

	int i, j, h, k, counter1, counter2;

	counter1 = 0;
	for (i = 0 ; i < d ; i++) {
		for (j = i + 1 ; j < d ; j++) {
			counter2 = 0;
			for (h = 0 ; h < d ; h++) {
				for (k = h + 1 ; k < d ; k++) {
					ret[ (n + d + counter1) + (n + d + counter2) * nrow ] = d4covdxjkxjlximxin(m, m, sigma, l, d, i, j, h, k);
					counter2++;
				}
			}
			counter1++;
		}
	}
}

/* Function which fills in the covariance entries between the non diagonal hesssian entries and the diagonal hessian entries */

void computeCovNonDiagHessianDiagHessian(double *ret, int n, int nrow, int d, double *l, double sigma, double *m) {

	int i, j, h, counter1;

	counter1 = 0;
	for (i = 0 ; i < d ; i++) {
		for (j = i + 1 ; j < d ; j++) {
			for (h = 0 ; h < d ; h++) {
				ret[ (n + d + counter1) + (n + d + d * (d - 1) / 2 + h) * nrow ] =
					d4covdxjkxjlximxin(m, m, sigma, l, d, i, j, h, h);
				ret[ (n + d + d * (d - 1) / 2 + h) + (n + d + counter1) * nrow ] =
					ret[ (n + d + counter1) + (n + d + d * (d - 1) / 2 + h) * nrow ];
			}
			counter1++;
		}
	}
}

/* Function which fills in the covariance entries between the non diagonal hesssian entries and the diagonal hessian entries */

void computeCovNonDiagHessianMinimum(double *ret, int n, int nrow, int d, double *l, double sigma, double *m) {

	int i, j, h, counter1;

	counter1 = 0;
	for (i = 0 ; i < d ; i++) {
		for (j = i + 1 ; j < d ; j++) {
			for (h = 0 ; h < d ; h++) {
				ret[ (n + d + counter1) + (n + d + d * (d - 1) / 2 + d) * nrow ] = d2covdxjkxjl(m, m, sigma, l, d, i, j);
				ret[ (n + d + d * (d - 1) / 2 + d) + (n + d + counter1) * nrow ] =
					ret[ (n + d + counter1) + (n + d + d * (d - 1) / 2 + d) * nrow ];
			}
			counter1++;
		}
	}
}

/********** WE COMPUTE THE COVARIANCES BETWEEN THE DIAGONAL HESSIAN AND EVERYTHING ELSE **********/

/* Function which fills in the covariance entries between the diagonal hesssian entries and the diagonal hessian entries */

void computeCovDiagHessianDiagHessian(double *ret, int n, int nrow, int d, double *l, double sigma, double *m) {

	int i, j;

	for (i = 0 ; i < d ; i++) {
		for (j = i ; j < d ; j++) {
			ret[ (n + d + d * (d - 1) / 2 + i) + (n + d + d * (d - 1) / 2 + j) * nrow ] =
				d4covdxjkxjlximxin(m, m, sigma, l, d, i, i, j, j);
			ret[ (n + d + d * (d - 1) / 2 + j) + (n + d + d * (d - 1) / 2 + i) * nrow ] =
				ret[ (n + d + d * (d - 1) / 2 + i) + (n + d + d * (d - 1) / 2 + j) * nrow ];
		}
	}
}

/* Function which fills in the covariance entries between the diagonal hesssian entries and minimum */

void computeCovDiagHessianMinimum(double *ret, int n, int nrow, int d, double *l, double sigma, double *m) {

	int i;

	for (i = 0 ; i < d ; i++) {
		ret[ (n + d + d * (d - 1) / 2 + i) + (n + d + d * (d - 1) / 2 + d) * nrow ] = d2covdxjkxjl(m, m, sigma, l, d, i, i);
		ret[ (n + d + d * (d - 1) / 2 + d) + (n + d + d * (d - 1) / 2 + i) * nrow ] =
			ret[ (n + d + d * (d - 1) / 2 + i) + (n + d + d * (d - 1) / 2 + d) * nrow ];
	}
}

/********** WE COMPUTE THE COVARIANCES BETWEEN THE MINIMUM AND EVERYTHING ELSE **********/

/* Function which fills in the covariance entries between the minimum and the minimum */

void computeMinimumMinimum(double *ret, int n, int nrow, int d, double *l, double sigma, double sigma0, double *m) {

	ret[ (n + d + d * (d - 1) / 2 + d) + (n + d + d * (d - 1) / 2 + d) * nrow ] = cov(m, m, sigma, l, d) + sigma0;
}

/********** WE COMPUTE THE COVARIANCES BETWEEN Xstar AND EVERYTHING ELSE **********/

/* Function which fills in the covariance entries between xstar and the n data points */

void computeCovXstarDataPoints(double *ret, double *Xstar, int nXstar,
	double *X, int n, int nrow, int d, double *l, double sigma, double sigma0) {

	int i, j, k;

	/* We allocate memory to store the data points */

	double *x1 = (double *) malloc(sizeof(double) * d);
	double *x2 = (double *) malloc(sizeof(double) * d);

	/* We compute the covariance function at the n available observations */

	for (i = 0 ; i <  nXstar ; i++) {
		for (j = 0 ; j < n ; j++) {

			/* We store the data points at which to evaluate the covariance function */

			for (k = 0 ; k < d ; k++) {
				x1[ k ] = Xstar[ i + nXstar * k ];
				x2[ k ] = X[ j + n * k ];
			}

			/* We evaluate the covariance function */

			ret[ i + j * nXstar ] = cov(x1, x2, sigma, l, d);
		}
	}

	free(x1); free(x2);
}

/* Function which fills in the covariance entries between xstar and the gradient at the minimum */

void computeCovXstarGradient(double *ret, double *Xstar, int nXstar, double *X, int n, int nrow, int d, double *l, double sigma, double *m) {

	int i, j, k;

	/* We allocate memory to store the data points */

	double *x1 = (double *) malloc(sizeof(double) * d);

	/* We evaluate the covariance function at the n available observations and d gradient variables */

	for (i = 0 ; i < nXstar ; i++) {
		for (j = 0 ; j < d ; j++) {

			/* We store the data points at which to evaluate the covariance function */

			for (k = 0 ; k < d ; k++)
				x1[ k ] = Xstar[ i + nXstar * k ];

			/* We evaluate the covariance function */

			ret[ i + (n + j) * nXstar ] = dcovdxjk(x1, m, sigma, l, d, j);
		}
	}

	free(x1);
}

/* Function which fills in the covariance entries between xstar and the non diagonal hessian elements */

void computeCovXstarNonDiagHessian(double *ret, double *Xstar, int nXstar, double *X, int n, int nrow, int d, double *l, double sigma, double *m) {

	int i, j, k, h, counter;

	/* We allocate memory to store the data points */

	double *x1 = (double *) malloc(sizeof(double) * d);

	for (i = 0 ; i < nXstar ; i++) {
		counter = 0;
		for (j = 0 ; j < d ; j++) {
			for (h = j + 1 ; h < d ; h++) {

				/* We store the data points at which to evaluate the covariance function */

				for (k = 0 ; k < d ; k++)
					x1[ k ] = Xstar[ i + nXstar * k ];

				/* We evaluate the covariance function */

				ret[ i + (n + d + counter) * nXstar ] = d2covdxjkxjl(x1, m, sigma, l, d, j, h);

				counter++;
			}
		}
	}

	free(x1);
}

/* Function which fills in the covariance entries between xstar and the diagonal hessian elements */

void computeCovXstarDiagHessian(double *ret, double *Xstar, int nXstar, double *X, int n, int nrow, int d, double *l, double sigma, double *m) {

	int i, j, k;

	/* We allocate memory to store the data points */

	double *x1 = (double *) malloc(sizeof(double) * d);

	for (i = 0 ; i < nXstar ; i++) {
		for (j = 0 ; j < d ; j++) {

			/* We store the data points at which to evaluate the covariance function */

			for (k = 0 ; k < d ; k++)
				x1[ k ] = Xstar[ i + nXstar * k ];

			/* We evaluate the covariance function */

			ret[ i + (n + d + d * (d - 1) / 2 + j) * nXstar ] = d2covdxjkxjl(x1, m, sigma, l, d, j, j);
		}
	}

	free(x1);
}

/* Function which fills in the covariance entries between xstar and the minimum */

void computeCovXstarMinimum(double *ret, double *Xstar, int nXstar, double *X, int n, int nrow, int d, double *l, double sigma, double *m) {

	int i, k;

	/* We allocate memory to store the data points */

	double *x1 = (double *) malloc(sizeof(double) * d);

	for (i = 0 ; i < nXstar ; i++) {

		/* We store the data points at which to evaluate the covariance function */

		for (k = 0 ; k < d ; k++)
			x1[ k ] = Xstar[ i + nXstar * k ];

		/* We evaluate the covariance function */

		ret[ i + (n + d + d * (d - 1) / 2 + d) * nXstar ] = cov(x1, m, sigma, l, d);
	}

	free(x1);
}


/*******************************************************************************
 * Modification to the original "compute the covariance matrix" code that can be
 * called externally.
 ******************************************************************************/

void se_cov(double *ret, double *X, int n, int d, double *m,
            double *l, double sigma, double sigma0)
{
    int nrow = n + d * d + d + 1;

	/********** WE COMPUTE THE COVARIANCES BETWEEN THE DATA POINTS AND EVERYTHING ELSE **********/

	/* We compute the covariances the n available observations */

	computeCovDataPointsDataPoints(ret, X, n, nrow, d, l, sigma, sigma0);

	/* We compute the covariances at the n available observations and the d gradient values at the minimum */

	computeCovDataPointsGradient(ret, X, n, nrow, d, l, sigma, m);

	/* We compute the covariances at the n available observations and the d * (d - 1) / 2 non diagonal entries of the hessian at the minimum */

	computeCovDataPointsNonDiagHessian(ret, X, n, nrow, d, l, sigma, m);

	/* We compute the covariances at the n available observations and the d diagonal entries of the hessian at the minimum */

	computeCovDataPointsDiagHessian(ret, X, n, nrow, d, l, sigma, m);

	/* We compute the covariances at the n available observations and the minimum */

	computeCovDataPointsMinimum(ret, X, n, nrow, d, l, sigma, m);

	/********** WE COMPUTE THE COVARIANCES BETWEEN THE GRADIENT AND EVERYTHING ELSE **********/

	/* We compute the covariances between the d gradient values and the d  gradient values */

	computeCovGradientGradient(ret, n, nrow, d, l, sigma, m);

	/* We compute the covariances between the d gradient values and the d * (d - 1) / 2 non diagonal entries of the hessian at the minimum */

	computeCovGradientNonDiagHessian(ret, n, nrow, d, l, sigma, m);

	/* We compute the covariances between the d gradient values and the d diagonal entries of the hessian at the minimum */

	computeCovGradientDiagHessian(ret, n, nrow, d, l, sigma, m);

	/* We compute the covariances between the d gradient values and the minimum */

	computeCovGradientMinimum(ret, n, nrow, d, l, sigma, m);

	/********** WE COMPUTE THE COVARIANCES BETWEEN THE NON DIAGONAL HESSIAN AND EVERYTHING ELSE **********/

	/* We compute the covariances between the d * (d - 1) / 2 non diagonal entries of the hessian */

	computeCovNonDiagHessianNonDiagHessian(ret, n, nrow, d, l, sigma, m);

	/* We compute the covariances between the d * (d - 1) / 2 non diagonal entries of the hessian and the diagonal entries */

	computeCovNonDiagHessianDiagHessian(ret, n, nrow, d, l, sigma, m);

	/* We compute the covariances between the d * (d - 1) / 2 non diagonal entries of the hessian and the minimum */

	computeCovNonDiagHessianMinimum(ret, n, nrow, d, l, sigma, m);

	/********** WE COMPUTE THE COVARIANCES BETWEEN THE DIAGONAL HESSIAN AND EVERYTHING ELSE **********/

	computeCovDiagHessianDiagHessian(ret, n, nrow, d, l, sigma, m);

	computeCovDiagHessianMinimum(ret, n, nrow, d, l, sigma, m);

	/********** WE COMPUTE THE COVARIANCES BETWEEN THE DIAGONAL HESSIAN AND EVERYTHING ELSE **********/

	computeMinimumMinimum(ret, n, nrow, d, l, sigma, sigma0, m);
}


void se_crosscov(double *ret, double *X, int n, int d, double *m,
                 double *l, double sigma, double sigma0,
                 double *Xstar, int nXstar)
{
    int nrow = (n + d + d * (d - 1) / 2 + d + 1);

    /********** WE COMPUTE THE COVARIANCES BETWEEN Xstar AND EVERYTHING ELSE **********/

    /* We compute the covariances the n available observations */

    computeCovXstarDataPoints(ret, Xstar, nXstar, X, n, nrow, d, l, sigma, sigma0);

    /* We compute the covariances at the n available observations and the d gradient values at the minimum */

    computeCovXstarGradient(ret, Xstar, nXstar, X, n, nrow, d, l, sigma, m);

    /* We compute the covariances at the n available observations and the d * (d - 1) / 2 non diagonal entries of the hessian at the minimum */

    computeCovXstarNonDiagHessian(ret, Xstar, nXstar, X, n, nrow, d, l, sigma, m);

    /* We compute the covariances at the n available observations and the d diagonal entries of the hessian at the minimum */

    computeCovXstarDiagHessian(ret, Xstar, nXstar, X, n, nrow, d, l, sigma, m);

    /* We compute the covariances at the n available observations and the minimum */

    computeCovXstarMinimum(ret, Xstar, nXstar, X, n, nrow, d, l, sigma, m);
}

