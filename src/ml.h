/*
 * ml.h
 *
 *  Created on: Sep 1, 2015
 *      Author: yaison
 */

#ifndef ML_H_
#define ML_H_



typedef struct Matrix{
	size_t m;
	size_t n;
	float* values;
}Matrix;

typedef struct LinearModel{
	size_t length;
	double bias;
	double* theta;
}LinearModel;

void train(Matrix* X, LinearModel* model, Matrix* y);
double abserr(Matrix* X, LinearModel* model, Matrix* Y);
void autostogdcent(Matrix* X, LinearModel* model, Matrix* y, double lambda);
void stogdcent(Matrix* X, LinearModel* model, Matrix* y, double alpha, double lambda,
		unsigned int batch, unsigned int steps);
void modrand(LinearModel* model);
LinearModel* modlinear(size_t length);
void modfree(LinearModel* mod);
double h(float* x, size_t n, double bias, double* theta);
double j(Matrix* X, LinearModel* model, Matrix* y, double lambda);
Matrix* mtrrange(Matrix* mtr, size_t fromIdx, size_t toIdx);
Matrix* mtrslct(Matrix* mtr, size_t startInc, size_t endExc);
Matrix* mtrxcl(Matrix* mtr, size_t col);
Matrix* mtrnew(size_t m, size_t n);
void mtrprint(Matrix* matrix);
void mtrfree(Matrix* matrix);
void copyd(double* to, double* from, size_t n);


#endif /* ML_H_ */
