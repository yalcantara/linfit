/*
 * uiutils.c
 *
 *  Created on: Sep 7, 2015
 *      Author: yaison
 */

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <dirent.h>

#include "utils.h"
#include "ui.h"


/*
 * This function reads and parse the contents of the file given by the
 * filePath parameter and construct the Data struct. In order to release
 * the Data pointer use datafree function.
 */
Data* datastep(char* filePath) {

	String* raw = ffull(filePath);
	if (raw) {
		//if raw isn't null, the ffull executed with no problems.
		
		//let us change '\' (windows) for '/'.
		for (int i = 0;; i++) {
			char c = filePath[i];
			if(c == '\0'){
				break;
			}
			if (c == '\\') {
				filePath[i] = '/';
			}
		}

		printf("Read file content %s.\n", filePath);
		printf("Building structures:\n");
		flush();
		Grid* g = gcreate(raw->value, ',');
		if (g) {
			//after the Grid struct is created, we no longer need the
			//raw pointer.
			strfree(raw);
			printf("Grid created.\n");
			Mapper* map = mapcreate(g);
			if (map) {

				printf("Mapper created.\n");
				//let's create a clone of the info struct
				GridInfo* info = ginfo(g, g->info->rows, g->info->columns);
				if (info) {

					printf("GridInfo created.\n");
					flush();
					//at this point we no longer need the Grid struct.

					pcolinf(info);
					pyinf(map, info);
					Data* d = malloc(sizeof(Data));
					d->info = info;
					d->map = map;
					d->matrix = mtrcreate(g, map);
					gfree(g);

					return d;
				}

				mapfree(map);
			} else {
				gfree(g);
			}
		} else {

			strfree(raw);
		}
	} else {
		printf("There was a problem with the file %s.\n", abs);
	}

	return NULL;
}


/*
 * This function checks the y column (last right) and determines if it's
 * numeric or a word column. In case it's a word column, it will delegate
 * to the wordtrain function, otherwise it will call the numerictrain function.
 */
void trainstep(Data* data) {

	Matrix* mtr = data->matrix;
	mtrshuffle(mtr);

	size_t cols = data->map->cols;
	size_t ycol = cols - 1;
	size_t words = data->info->words[ycol];
	if (words > 0) {
		//as long as there is at least 1 word, then
		//it is considered a word column.
		wordtrain(data);
	} else {
		numerictrain(data);
	}
}


/*
 * This function asumes that the y column (last right), is numeric and thus
 * creates the X matrix from [0, n-1) where n is the number of columns of the
 * data's matrix.
 */
void numerictrain(Data* data) {
	Matrix* mtr = data->matrix;
	size_t n = mtr->n;

	printf("====================================================\n");
	Matrix* X = mtrslct(mtr, 0, n - 1);
	size_t xn = X->n;

	LinearModel* model = modlinear(xn);

	Matrix* y = mtrslct(mtr, n - 1, n);
	//once the X, model, and y structures are built, training is as
	//simple as calling the dotrain function.
	dotrain(X, model, y);

	mtrfree(X);
	mtrfree(y);
	modfree(model);
	printf("====================================================\n\n");
	
}

/*
 * This function asumes that the y column (last right), is a word column
 * which means it may have multiple classes. In that case, we need to
 * train for each class and compute it's own cost value.
 */
void wordtrain(Data* data) {

	Matrix* mtr = data->matrix;
	char*** map = data->map->map;
	size_t* sizes = data->map->sizes;
	size_t* missing = data->info->missing;

	size_t cols = data->map->cols;
	size_t ycol = cols - 1;
	size_t ycount = data->map->sizes[ycol];
	size_t ystart = mtrcols(map, sizes, missing, ycol);

	//it doesn't matter the the type of class of the y column, the X matrix
	//will be the same for all classes. That is why we can safely create
	//the X matrix at this point and reuse it for each class.
	Matrix* X = mtrslct(mtr, 0, ystart);
	size_t xn = X->n;

	for (size_t i = 0; i < ycount; i++) {
		char* yval = map[ycol][i];
		//since each class has a corresponding column in the Data's matrix,
		//we need to compute that index. yidx is the class target idx in the
		//Data's matrix.
		size_t yidx = tomtrcol(map, sizes, missing, ycol, yval);

		printf("====================================================\n");
		printf("y: %s\n\n", yval);
		
		//the +1 is just for indexing purpose: [yidx, yidx+1)
		Matrix* y = mtrslct(mtr, yidx, yidx + 1);

		LinearModel* model = modlinear(xn);

		//once the X, model, and y structures are built, training is as
		//simple as calling the dotrain function.
		dotrain(X, model, y);

		mtrfree(y);
		modfree(model);
		printf("====================================================\n\n");
	}
	
	mtrfree(X);
}

void dotrain(Matrix* X, LinearModel* model, Matrix* y) {
	size_t xn = X->n;

	double jbefore = j(X, model, y, 0);
	printf("Before j: %12.8f\n", jbefore);
	printf("Training... please wait.\n");
	flush();
	train(X, model, y);
	double jafter = j(X, model, y, 0);
	printf("After  j: %12.8f\n", jafter);

	printf("\nSome examples\n\n");

	for (int i = 0; i < 10; i++) {
		float* x = X->values + i * xn;
		double ans = h(x, xn, model->bias, model->theta);
		printf("%8.4f  ->  %8.4f\n", y->values[i], ans);
	}

}

void datafree(Data* data) {
	if (data) {
		if (data->info) {
			ginfofree(data->info);
		}

		if (data->map) {
			mapfree(data->map);
		}

		if (data->matrix) {
			mtrfree(data->matrix);
		}

		free(data);
	}
}

void pyinf(Mapper* map, GridInfo* info) {
	size_t ycol = (map->cols) - 1;
	printf("\n");
	if (info->words[ycol] > 0) {
		printf("The type of the y column is: word.\n");
		printf("y values: ");
		size_t yvals = map->sizes[ycol];
		for (size_t i = 0; i < yvals; i++) {
			printf("'%s'", map->map[ycol][i]);
			if (i + 1 < yvals) {
				printf(", ");
			}
		}
		printf(".\n");

	} else if (info->discrete[ycol]) {
		printf("The type of the y column is: int.\n");
	} else {
		printf("The type of the y column is: float.\n");
	}
}

/*
 * Prints information of the GridInfo struct.
 */
void pcolinf(GridInfo* info) {
	size_t cols = info->columns;
	printf("\nColumn details:\n\n");
	printf(
			"------------------------------------------------------------------------------------------------------------\n");
	printf(
			"|  Index  |   Type   |   Count   |  Missing  |     Max     |     Min     |     Mean     |   Standard Dev   |\n");
	printf(
			"------------------------------------------------------------------------------------------------------------\n");
	flush();
	for (size_t j = 0; j < cols; j++) {
		short numeric = info->words[j] == 0;

		size_t missing = info->missing[j];
		printf("| %6d ", (j + 1));
		flush();
		if (numeric) {
			size_t count = info->numbers[j];
			short discrete = info->discrete[j];
			float max = info->max[j];
			float min = info->min[j];
			float mean = info->mean[j];
			float stdev = info->stdev[j];

			if (discrete) {
				printf(
						" |   int    | %9d | %9d |  %9.0f  |  %9.0f  | %12.2f | %16.2f |",
						count, missing, max, min, mean, stdev);
			} else {
				printf(
						" |   float  | %9d | %9d |  %9.2f  |  %9.2f  | %12.2f | %16.2f |",
						count, missing, max, min, mean, stdev);
			}
		} else {
			size_t count = 0;
			count += info->words[j];
			count += info->numbers[j];
			printf(
					" |   word   | %9d | %9d |         -   |         -   |          -   |              -   |",
					count, missing);
		}

		printf("\n");
	}
	printf(
			"------------------------------------------------------------------------------------------------------------\n");
}

void pfiles(char** fileNames, char* buf, size_t buflen, size_t* l, short* found) {

	if (getcwd(buf, buflen) != NULL) {
		DIR* dir;

		*l = strlen(buf);
		char* fpath = "/files/";
		strcpy(buf + *l, fpath);
		*(l) += 7;

		if ((dir = opendir(buf)) != NULL) {
			struct dirent* ent;

			while ((ent = readdir(dir)) != NULL) {
				pdir(ent, fileNames, found);
			}

			closedir(dir);
		}
	}
}


/*
 * Prints the *.data files under the given directory parameter ent. For each,
 * it will also print it's index (starting from 1). 
 */
void pdir(struct dirent* ent, char** fileNames, short* found) {
	if (ent->d_type != DT_REG) {
		return;
	}

	char* name = ent->d_name;
	char* p = strchr(name, '.');

	//lowercmp, cuz we don't really care cases.
	if (lowercmp(p, ".data") == 1) {
		if (*found == 0) {
			printf("Choose one of these files:\n\n");
		}

		char* cname = cnew(strlen(name));
		strcpy(cname, name);

		fileNames[*found] = cname;
		(*found)++;
		printf("%4d - %s\n", *found, name);
	}
}



void mtrprint(Matrix* matrix) {
	mtrlmprint(matrix, matrix->m);
}

/*
 * Matrix Limited Print
 * 
 * Prints the content of a matrix starting from the first row up to the
 * 'rows' parameter.
 */
void mtrlmprint(Matrix* matrix, size_t rows) {
	size_t n = matrix->n;
	float* values = matrix->values;

	for (size_t i = 0; i < rows; i++) {
		printf("%5d  ", (i + 1));
		for (size_t j = 0; j < n; j++) {
			printf("%6.2f  ", values[i * n + j]);
		}
		printf("\n");
	}
}
