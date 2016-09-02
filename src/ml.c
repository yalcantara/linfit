/*
 * ml.c
 *
 *  Created on: Sep 1, 2015
 *      Author: yaison
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "ml.h"
#include <pthread.h>

void train(Matrix* X, LinearModel* model, Matrix* y) {

	modrand(model);

	size_t m = X->m;
	size_t n = X->n;
	size_t trainIdx = floor(m * 0.7);

	Matrix* xtrain = mtrrange(X, 0, trainIdx);
	Matrix* ytrain = mtrrange(y, 0, trainIdx);

	Matrix* xtest = mtrrange(X, trainIdx, m);
	Matrix* ytest = mtrrange(y, trainIdx, m);

	double originalBias = model->bias;
	double originalTheta[n];
	copyd(originalTheta, model->theta, n);

	double lwlambda = 0;
	double lwbias = model->bias;
	double lwtheta[n];
	copyd(lwtheta, model->theta, n);

	double lwj = j(xtest, model, ytest, 0);

	double lambdas[] = { 0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30,
			100, 300 };
	size_t length = sizeof(lambdas) / sizeof(double);

	for (int i = 0; i < length; i++) {
		double lambda = lambdas[i];
		autostogdcent(xtrain, model, ytrain, lambda);
		double newj = j(xtest, model, ytest, 0);
		if (newj < lwj) {
			lwbias = model->bias;
			copyd(lwtheta, model->theta, n);
			lwlambda = lambda;
			lwj = newj;
		}

		if (i + 1 < length) {
			model->bias = originalBias;
			copyd(model->theta, originalTheta, n);
		}
	}

	mtrfree(xtrain);
	mtrfree(ytrain);
	mtrfree(xtest);
	mtrfree(ytest);
	
	printf("lambda: %f\n", lwlambda);
	model->bias = lwbias;
	copyd(model->theta, lwtheta, n);

}

void autostogdcent(Matrix* X, LinearModel* model, Matrix* y, double lambda) {

	size_t n = X->n;
	size_t m = X->m;

	unsigned int batch;
	if (m < 10) {
		batch = 1;
	} else if (m < 20) {
		batch = 4;
	} else if (m < 50) {
		batch = 10;
	} else if (m < 200) {
		batch = 20;
	} else {
		batch = 50;
	}

	double lwbias = model->bias;
	double lwtheta[n];
	copyd(lwtheta, model->theta, n);

	double crtj = j(X, model, y, lambda);

	double alpha = 0.1;
	unsigned int steps = y->m * 1000;
	for (int i = 0; i < 10; i++) {
		stogdcent(X, model, y, alpha, lambda, batch, steps);
		double newj = j(X, model, y, lambda);
		if (newj < crtj) {
			crtj = newj;
			lwbias = model->bias;
			copyd(lwtheta, model->theta, n);
		} else {
			alpha /= 10;
			model->bias = lwbias;
			copyd(model->theta, lwtheta, n);
		}
	}

	model->bias = lwbias;
	copyd(model->theta, lwtheta, n);
}

void stogdcent(Matrix* X, LinearModel* model, Matrix* y, double alpha,
		double lambda, unsigned int batch, unsigned int steps) {
	size_t n = X->n;
	size_t m = X->m;

	double* theta = model->theta;
	float* ans = y->values;
	float* vals = X->values;
	size_t tl = n + 1;

	double* batchAvg = malloc(sizeof(double) * tl);
	double bias = model->bias;

	size_t mainCounter = 0;

	while (mainCounter < steps) {
		mainCounter++;

		for (size_t j = 0; j < tl; j++) {
			batchAvg[j] = 0.0;
		}

		size_t counter = 0;
		size_t i = 0;
		while (counter < batch) {
			counter++;

			float* x = vals + i * n;
			float yi = ans[i];

			double hi = h(x, n, bias, theta) - yi;
			batchAvg[0] += hi;

			for (size_t j = 0; j < n; j++) {
				batchAvg[j + 1] += hi * x[j];
			}

			i++;
			if (i >= m) {
				i = 0;
			}
		}

		if (batch > 1) {
			for (size_t j = 0; j < tl; j++) {
				batchAvg[j] /= batch;
			}
		}

		//assign
		bias = bias - alpha * batchAvg[0];
		for (size_t j = 0; j < n; j++) {
			theta[j] = theta[j] * (1 - alpha * lambda)
					- alpha * batchAvg[j + 1];
		}

		model->bias = bias;
	}

	free(batchAvg);
}

double h(float* x, size_t n, double bias, double* theta) {

	double ans = bias;

	for (size_t i = 0; i < n; i++) {
		ans += x[i] * theta[i];
	}

	return ans;
}

double j(Matrix* X, LinearModel* model, Matrix* y, double lambda) {

	size_t m = X->m;
	size_t n = X->n;

	float* vals = X->values;

	double bias = model->bias;
	double* theta = model->theta;

	float* ans = y->values;

	double sum = 0.0;

	for (size_t i = 0; i < m; i++) {
		float* x = vals + i * n;
		float yi = ans[i];
		double hx = h(x, n, bias, theta);

		sum += pow(hx - yi, 2);
	}

	if (lambda != 0.0) {
		double regsum = 0.0;
		for (size_t j = 0; j < n; j++) {
			regsum += pow(theta[j], 2);
		}

		sum += lambda * regsum;
	}

	sum = 1.0 / (2 * m) * sum;

	return sum;
}

Matrix* mtrrange(Matrix* mtr, size_t fromIdx, size_t toIdx) {
	if (toIdx <= fromIdx || fromIdx < 0) {
		fflush(stdout);
		fprintf(stderr, "Invalid arguments. fromIdx: %d, toIdx: %d.\n", fromIdx,
				toIdx);
		fflush(stderr);
		return NULL;
	}

	size_t m = toIdx - fromIdx;
	size_t n = mtr->n;

	float* src = mtr->values;

	Matrix* matrix = mtrnew(m, n);
	float* values = matrix->values;

	for (size_t i = 0; i < m; i++) {
		for (size_t j = 0; j < n; j++) {
			values[i * n + j] = src[(fromIdx + i) * n + j];
		}
	}

	return matrix;
}

Matrix* mtrslct(Matrix* mtr, size_t startInc, size_t endExc) {
	if (endExc <= startInc || startInc < 0) {
		fflush(stdout);
		fprintf(stderr, "Invalid arguments. startInc: %d, endExc: %d.",
				startInc, endExc);
		fflush(stderr);
		return NULL;
	}

	size_t m = mtr->m;
	size_t n = endExc - startInc;

	float* src = mtr->values;
	size_t srcn = mtr->n;

	Matrix* matrix = mtrnew(m, n);
	float* values = matrix->values;

	for (size_t i = 0; i < m; i++) {
		for (size_t j = 0; j < n; j++) {
			values[i * n + j] = src[i * srcn + startInc + j];
		}
	}

	return matrix;
}

LinearModel* modlinear(size_t length) {
	LinearModel* m = malloc(sizeof(LinearModel));
	m->bias = 1.0;
	m->length = length;
	m->theta = calloc(length, sizeof(double));

	return m;
}

void modrand(LinearModel* model) {
	size_t l = model->length;
	model->bias = 1.0;
	for (size_t i = 0; i < l; i++) {
		//range [-1, 1]. 
		model->theta[i] = -1 + 2 * ((double) rand() / (double) (RAND_MAX));
	}
}

void modfree(LinearModel* mod) {
	if (mod) {
		if (mod->theta) {
			free(mod->theta);
		}

		free(mod);
	}
}

Matrix* mtrxcl(Matrix* mtr, size_t col) {
	size_t m = mtr->m;
	size_t n = mtr->n - 1;

	float* src = mtr->values;

	Matrix* matrix = mtrnew(m, n);
	float* dest = matrix->values;

	for (size_t i = 0; i < m; i++) {
		for (size_t j = 0; j < n; j++) {
			if (j < col) {
				dest[i * n + j] = src[i * (n + 1) + j];
			} else {
				dest[i * n + j] = src[i * (n + 1) + j + 1];
			}
		}
	}

	return matrix;
}

Matrix* mtrnew(size_t m, size_t n) {

	Matrix* matrix = malloc(sizeof(Matrix));
	matrix->m = m;
	matrix->n = n;
	matrix->values = calloc(m * n, sizeof(float));

	return matrix;
}

void mtrfree(Matrix* matrix) {
	if (matrix) {
		if (matrix->values) {
			free(matrix->values);
		}

		free(matrix);
	}
}

void copyd(double* to, double* from, size_t n) {
	for (size_t i = 0; i < n; i++) {
		to[i] = from[i];
	}
}
