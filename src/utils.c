#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <limits.h>
#include <string.h>
#include <ctype.h>
#include <math.h>

#include "utils.h"
#include "ml.h"


/*
 * Compares two strings by converting each character to lower case.
 * The conversion is done internally and the parameters are not modified. 
 */
unsigned int lowercmp(char* str, char* other) {

	for (size_t i = 0;; i++) {
		char s = str[i];
		char o = other[i];
		if (s == '\0' && o == '\0') {
			//at this point we have reached the end of the string
			//and both of them have the same length. So, they are equals.
			return 1;
		}

		if (s == '\0' || o == '\0') {
			return -i;
		}
		
		if (tolower(s) != tolower(o)) {
			return -i;
		}
	}

	return 0;
}

/*
 * Shuffles the rows of a given matrix.
 */
void mtrshuffle(Matrix* matrix) {

	float* values = matrix->values;
	size_t m = matrix->m;
	size_t n = matrix->n;

	float* buffer = malloc(sizeof(float) * n);
	for (size_t i = 0; i < m; i++) {
		size_t idxTo = (size_t) ((m - 1) * (rand() / (double) RAND_MAX));
		mtrswap(values, n, buffer, i, idxTo);
	}
	free(buffer);
}


/*
 * Swaps two rows.
 */
void mtrswap(float* values, size_t n, float* buffer, size_t idxFrom,
		size_t idxTo) {
	if (idxFrom == idxTo) {
		return;
	}

	//first copies the row at idxTo row in to a buffer.
	for (size_t j = 0; j < n; j++) {
		buffer[j] = values[idxTo * n + j];
	}

	//then writes the values of the row at idxFrom to the row at idxTo
	for (size_t j = 0; j < n; j++) {
		values[idxTo * n + j] = values[idxFrom * n + j];
	}

	//and last, writes the values of the buffer to the row at idxFrom
	for (size_t j = 0; j < n; j++) {
		values[idxFrom * n + j] = buffer[j];
	}
}

Matrix* mtrcreate(Grid* g, Mapper* mapper) {

	size_t m = g->info->rows;

	size_t* missing = g->info->missing;

	size_t* sizes = mapper->sizes;
	char*** map = mapper->map;

	size_t cols = mapper->cols;

	size_t n = mtrcols(map, sizes, missing, cols);

	Matrix* matrix = mtrnew(m, n);
	float* values = matrix->values;

	size_t rows = g->info->rows;
	char*** body = g->body;
	for (size_t i = 0; i < rows; i++) {
		for (size_t j = 0; j < cols; j++) {
			char* val = body[i][j];
			size_t mtrcol = tomtrcol(map, sizes, missing, j, val);
			if (mtrcol == -1) {
				fprintf(stderr, "Could not create Matrix");
				fflush(stderr);
				return NULL;
			}

			size_t words = g->info->words[j];
			if (val == NULL || words > 0) {
				values[i * n + mtrcol] = 1;
			} else {
				float mean = g->info->mean[j];
				float stdev = g->info->stdev[j];

				float v = (atof(val) - mean) / stdev;
				values[i * n + mtrcol] = v;
			}
		}
	}

	return matrix;
}

void fillstdev(GridInfo* info, Grid* g) {

	size_t cols = info->columns;
	float* means = info->mean;
	float* stdevs = info->stdev;

	for (size_t i = 0; i < cols; i++) {
		float mean;
		float stdev;
		onlinestdev(info, g, i, &mean, &stdev);
		means[i] = mean;
		stdevs[i] = stdev;
	}
}

void onlinestdev(GridInfo* info, Grid* g, size_t col, float* mean, float* stdev) {

	size_t m = info->rows;
	if (m < 2 || info->words[col] > 0) {
		*mean = NAN;
		*stdev = NAN;
		return;
	}

	char*** body = g->body;

	double n = 0;
	double _mean = 0.0;
	double m2 = 0.0;
	for (size_t i = 0; i < m; i++) {
		char* val = body[i][col];

		char* tailptr;
		double x = strtod(val, &tailptr);
		if (val == tailptr) {
			//not a number
		} else {

			n++;

			double delta = x - _mean;
			_mean += delta / n;
			m2 += delta * (x - _mean);
		}
	}

	*mean = (float) _mean;
	*stdev = (float) m2 / (n - 1);
}

size_t tomtrcol(char*** map, size_t* sizes, size_t* missing, size_t col,
		char* val) {
	size_t left = mtrcols(map, sizes, missing, col);
	if (map[col] == NULL) {
		if (missing[col] == 0) {
			return left;
		}

		if (val == NULL) {
			return left;
		}

		return left + 1;
	}

	size_t l = sizes[col];
	for (size_t i = 0; i < l; i++) {
		char* crt = map[col][i];

		if (crt == NULL && val == NULL) {
			return left + i;
		}

		if ((crt != NULL && val != NULL) && strcmp(crt, val) == 0) {
			return left + i;
		}
	}

	fprintf(stderr, "could not convert to matrix "
			"column for grid column %d and value %s", col, val);
	fflush(stderr);
	return -1;
}

size_t mtrcols(char*** map, size_t* sizes, size_t* missing, size_t col) {
	size_t n = 0;
	for (size_t j = 0; j < col; j++) {
		if (map[j] == NULL) {
			//numeric type
			if (missing[j] == 0) {
				n++;
			} else {
				n += 2;
			}
		} else {
			n += sizes[j];
		}
	}
	return n;
}

Mapper* mapcreate(Grid* g) {

	size_t n = g->info->columns;

	size_t* sizes = calloc(n, sizeof(size_t));
	char*** map = calloc(n, sizeof(char**));

	for (size_t j = 0; j < n; j++) {
		size_t numbers = g->info->numbers[j];
		size_t words = g->info->words[j];

		if (numbers == 0 && words == 0) {
			//empty column
			sizes[j] = 0;
			map[j] = NULL;
		} else if (numbers == 0 && words > 0) {
			//all words
			size_t l;
			char** arr = struniq(g, j, &l);
			sizes[j] = l;
			map[j] = arr;

		} else if (numbers > 0 && words == 0) {
			//all numbers
			sizes[j] = 0;
			map[j] = NULL;
		} else {
			//mixed, treated as words
			size_t l;
			char** arr = struniq(g, j, &l);
			sizes[j] = l;
			map[j] = arr;
		}
	}

	Mapper* mapper = malloc(sizeof(Mapper));
	mapper->cols = n;
	mapper->sizes = sizes;
	mapper->map = map;

	return mapper;
}

void mapfree(Mapper* mapper) {
	if (mapper) {
		char*** map = mapper->map;
		size_t cols = mapper->cols;
		size_t* sizes = mapper->sizes;

		if (map) {
			for (size_t i = 0; i < cols; i++) {
				size_t size = sizes[i];
				for (size_t j = 0; j < size; j++) {
					free(map[i][j]);
				}
				free(map[i]);
			}
		}

		if (sizes) {
			free(sizes);
		}

		free(mapper);
	}
}

GridInfo* ginfo(Grid* g, size_t rows, size_t cols) {

	char*** body = g->body;

	GridInfo* info = malloc(sizeof(GridInfo));
	info->max = malloc(sizeof(float) * cols);
	info->min = malloc(sizeof(float) * cols);
	info->mean = malloc(sizeof(float) * cols);
	info->stdev = malloc(sizeof(float) * cols);
	info->discrete = malloc(sizeof(short) * cols);
	info->missing = malloc(sizeof(size_t) * cols);
	info->numbers = malloc(sizeof(size_t) * cols);
	info->words = malloc(sizeof(size_t) * cols);

	for (int j = 0; j < cols; j++) {
		short discrete;
		float max = NAN;
		float min = NAN;
		size_t missing = 0;
		size_t numbers = 0;
		size_t words = 0;

		size_t discreteCount = 0;
		for (int i = 0; i < rows; i++) {
			char* val = body[i][j];
			if (val == NULL) {
				missing++;
			} else {
				char* end;
				float num = strtof(val, &end);
				if (val == end) {
					//not a number
					words++;
				} else {
					if (ceil(num) == num) {
						discreteCount++;
					}
					max = higher(max, num);
					min = lower(min, num);
					numbers++;
				}
			}
		}

		if (numbers > 0) {
			discrete = discreteCount == numbers;
		}

		info->discrete[j] = discrete;
		info->max[j] = max;
		info->min[j] = min;
		info->missing[j] = missing;
		info->numbers[j] = numbers;
		info->words[j] = words;

	}
	info->rows = rows;
	info->columns = cols;

	fillstdev(info, g);

	return info;
}

Grid* gcreate(char* raw, char d) {
	ssize_t lines = cfind(raw, '\n');
	if (lines == -1) {
		fflush(stdout);
		fprintf(stderr, "The file has no \n char.");
		fflush(stderr);
		return NULL;
	}

	size_t cols = ccount(raw, lines, d);
	size_t rows = ccount(raw, -1, '\n');
	cols++;
	rows++;
	
	if (cols < 2) {
		fflush(stdout);
		fprintf(stderr, "Error creating Grid. Less than 2 columns.");
		fflush(stderr);
		return NULL;
	}

	if (rows < 10) {
		fflush(stdout);
		fprintf(stderr, "Error creating Grid. Less than 10 rows.");
		fflush(stderr);
		return NULL;
	}

	Grid* g = malloc(sizeof(Grid));
	g->info = NULL;
	char*** body = calloc(rows, sizeof(char**));
	g->body = body;

	size_t start = 0;
	char* p = raw;
	for (int i = 0; i < rows; i++) {
		char** row = calloc(cols, sizeof(char*));

		for (int j = 0; j < cols; j++) {
			p = p + start;

			if (p[0] == '\n' && j == 0) {
				//at this point there is an empty line.
				//we are going to stop there

				fflush(stdout);
				fprintf(stderr, "\nBlank line detected at line %d.\n", (i + 1));
				fflush(stderr);
				gpartialfree(g, i - 1, cols);
				free(row);
				return NULL;
			}

			ssize_t l;
			if (j + 1 < cols) {
				l = cfind(p, d);
			} else if (i + 1 < rows) {
				l = cfind(p, '\n');
			} else {
				l = strlen(p);
			}

			if (l == -1) {
				fflush(stdout);
				fprintf(stderr, "\nInvalid row. Row %d, detected at col %d.\n",
						(i + 1), (j + 1));
				fflush(stderr);
				gpartialfree(g, i - 1, cols);
				free(row);
				return NULL;
			}

			size_t tl = triml(p, l);
			if (tl > 0) {
				char* col = cnew(tl);
				trim(p, l, col, tl);

				row[j] = col;
			} else {
				row[j] = NULL;
			}
			start = l + 1;
		}

		size_t allNull = 1;
		for (int j = 0; j < cols; j++) {
			if (row[j] != NULL) {
				allNull = 0;
				break;
			}
		}

		if (allNull) {
			fflush(stdout);
			fprintf(stderr,
					"\nThere is a problem with the file, got all column values null for row %d.\n",
					(i + 1));
			fflush(stderr);
			gpartialfree(g, i - 1, cols);
			free(row);
			return NULL;
		}

		body[i] = row;
	}

	g->info = ginfo(g, rows, cols);

	return g;
}

void gprint(Grid* g) {
	size_t m = g->info->rows;
	size_t n = g->info->columns;
	char*** values = g->body;

	for (size_t i = 0; i < m; i++) {
		printf("%5d  ", (i + 1));
		for (size_t j = 0; j < n; j++) {
			printf("%5s  ", values[i][j]);
		}
		printf("\n");
	}
}

char** struniq(Grid* g, size_t col, size_t* destLength) {

	char*** body = g->body;
	size_t m = g->info->rows;

	char** temp = malloc(sizeof(char*) * m);

	size_t assigned = 0;
	for (int i = 0; i < m; i++) {
		char* crt = body[i][col];

		short found = 0;
		for (int j = 0; j < assigned; j++) {
			char* other = temp[j];
			if (crt == NULL && other == NULL) {
				found = 1;
				break;
			}

			if (crt == NULL || other == NULL) {
				continue;
			}

			if (strcmp(crt, other) == 0) {
				found = 1;
				break;
			}
		}

		if (!found) {
			if (crt) {
				size_t l = strlen(crt);
				char* copy = cnew(l);
				strcpy(copy, crt);
				temp[assigned] = copy;
				assigned++;
			} else {
				temp[assigned] = NULL;
				assigned++;
			}
		}
	}

	char** uniq = malloc(sizeof(char*) * assigned);

	for (int i = 0; i < assigned; i++) {
		uniq[i] = temp[i];
	}

	free(temp);
	*destLength = assigned;
	return uniq;
}

size_t triml(char* src, size_t length) {
	ssize_t right = cnotfind(src, ' ');
	if (right == -1) {
		return 0;
	}

	ssize_t left = cnotfindr(src, length, ' ');
	if (left == -1) {
		return 0;
	}

	return left - right + 1;
}

size_t trim(char* src, size_t srcLength, char* dest, size_t destLength) {

	ssize_t right = cnotfind(src, ' ');
	if (right == -1) {
		return 0;
	}

	size_t left = cnotfindr(src, srcLength, ' ');
	if (left == -1) {
		return 0;
	}

	left++;

	size_t j = 0;
	for (size_t i = right; i < left; i++) {
		dest[j] = src[i];
		j++;
	}

	return j;
}

short blank(char* str, size_t length) {
	for (size_t i = 0; i < length; i++) {
		if (str[i] != ' ' || str[i] != '\n') {
			return 0;
		}
	}

	return 1;
}

float higher(float a, float b) {
	if (isnan(a)) {
		if (isnan(b)) {
			return NAN;
		}

		return b;
	}

	if (isnan(b)) {
		return a;
	}

	if (a > b) {
		return a;
	}

	return b;
}

float lower(float a, float b) {
	if (isnan(a)) {
		if (isnan(b)) {
			return NAN;
		}

		return b;
	}

	if (isnan(b)) {
		return a;
	}

	if (a < b) {
		return a;
	}

	return b;
}

size_t ccount(char* str, size_t end, char c) {
	if (end == -1) {
		end = LONG_MAX;
	}

	size_t sum = 0;
	char crt;
	for (size_t i = 0; i < end; i++) {
		crt = str[i];
		if (crt == 0) {
			return sum;
		}

		if (crt == c) {
			sum++;
		}
	}

	return sum;
}

void flush() {
	fflush(stdout);
}

ssize_t cfind(char* str, char c) {

	char crt;
	for (size_t i = 0;; i++) {
		crt = str[i];
		if (crt == '\0') {
			return -1;
		}

		if (crt == c) {
			return i;
		}
	}

	return -1;
}

ssize_t cnotfind(char* str, char c) {

	char crt;
	for (size_t i = 0;; i++) {
		crt = str[i];
		if (crt == '\0') {
			return -1;
		}

		if (crt != c) {
			return i;
		}
	}

	return -1;
}

ssize_t cnotfindr(char* str, size_t length, char c) {
	char crt;
	for (size_t i = length - 1; i >= 0; i--) {
		crt = str[i];
		if (crt == 0) {
			return -1;
		}

		if (crt != c) {
			return i;
		}
	}

	return -1;
}

String* ffull(char* path) {
	FILE* f = fopen(path, "r");

	if (f == NULL) {
		return NULL;
	}

	char c;

	size_t count = 0;
	while ((c = fgetc(f))) {
		if (c == EOF) {
			break;
		}
		count++;
	}

	char* content = cnew(count);
	rewind(f);
	size_t r = fread(content, sizeof(char), count, f);

	fclose(f);

	String* s = strnew(content, r);
	return s;
}

void strprintln(String* str) {
	printf(str->value);
	printf("\n");
	fflush(stdout);
}

void ginfofree(GridInfo* info) {
	if (info) {

		if (info->max)
			free(info->max);

		if (info->min)
			free(info->min);

		if (info->mean)
			free(info->mean);

		if (info->stdev)
			free(info->stdev);

		if (info->discrete)
			free(info->discrete);

		if (info->numbers)
			free(info->numbers);

		if (info->words)
			free(info->words);

		if (info->missing)
			free(info->missing);

		free(info);
	}
}

void gpartialfree(Grid* g, size_t rows, size_t cols) {
	if (g) {

		char*** body = g->body;
		if (body) {

			for (size_t i = 0; i < rows; i++) {
				if (body[i])
					for (size_t j = 0; j < cols; j++) {

						if (body[i][j]) {
							free(body[i][j]);
						}

					}
				free(body[i]);
			}

			free(body);
		}

		ginfofree(g->info);

		free(g);
	}
}

void gfree(Grid* g) {
	if (g) {

		char*** body = g->body;
		if (body) {
			if (g->info) {
				size_t m = g->info->rows;
				size_t n = g->info->columns;
				for (size_t i = 0; i < m; i++) {
					if (body[i]) {
						for (size_t j = 0; j < n; j++) {
							if (body[i][j]) {
								free(body[i][j]);
							}
						}
						free(body[i]);

					}
				}
			}
			free(body);

		}

		ginfofree(g->info);
		free(g);
	}
}

void strfree(String* str) {
	free((char*) str->value);
	free(str);
}

char* cnew(size_t l) {
	char* s = malloc(sizeof(char) * (l + 1));
	s[l] = 0;
	return s;
}

String* strnew(char* value, size_t length) {
	String* s = malloc(sizeof(String));
	s->length = length;
	s->value = value;
	return s;
}
