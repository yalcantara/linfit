/*
 * utils.h
 *
 *  Created on: Aug 23, 2015
 *      Author: yaison
 */

#ifndef UTILS_H_
#define UTILS_H_

#include <sys/types.h>
#include "ml.h"


typedef struct String{
	size_t length;
	char* value;
}String;

typedef struct GridInfo{
	float* max;
	float* min;
	float* mean;
	float* stdev;
	
	short* discrete;
	size_t* numbers;
	size_t* words;
	size_t* missing;
	
	size_t rows;
	size_t columns;
}GridInfo;

typedef struct Grid{
	
	GridInfo* info;
	char*** body;
}Grid;

typedef struct Mapper{
	size_t cols;
	size_t* sizes;
	char*** map;
}Mapper;


unsigned int lowercmp(char* str, char* other);
void mtrshuffle(Matrix* matrix);
void mtrswap(float* values, size_t n, float* buffer, size_t idxFrom, size_t idxTo);

void fillstdev(GridInfo* info, Grid* g);
void onlinestdev(GridInfo* info, Grid* g, size_t col, float* mean, float* stdev);

Matrix* mtrcreate(Grid* g, Mapper* mapper);
void gprint(Grid* g);
size_t tomtrcol(char*** map, size_t* sizes, size_t* missing, size_t col, char* val);
size_t mtrcols(char*** map, size_t* sizes, size_t* missing, size_t cols);

Mapper* mapcreate(Grid* g);
void mapfree(Mapper* mapper);
char** struniq(Grid* g, size_t col, size_t* destLength);
size_t triml(char* src, size_t srcLength);
size_t trim(char* src, size_t srcLength, char* dest, size_t destLength);
short blank(char* str, size_t length);
float higher(float a, float b);
float lower(float a, float b);

GridInfo* ginfo(Grid* g, size_t rows, size_t cols);
void ginfofree(GridInfo* info);
Grid* gcreate(char* raw, char d);

void gfree(Grid* g);
void gpartialfree(Grid* g, size_t rows, size_t cols);

size_t ccount(char* str, size_t end, char c);
void flush();
ssize_t cfind(char* str, char c);
ssize_t cnotfind(char* str, char c);
ssize_t cnotfindr(char* str, size_t length, char c);
String* ffull(char* path);
void strprintln(String* str);
char* cnew(size_t l);


void strfree(String* str);
String* strnew(char* value, size_t length);




#endif /* UTILS_H_ */
