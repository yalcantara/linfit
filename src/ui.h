/*
 * ui.h
 *
 *  Created on: Sep 7, 2015
 *      Author: yaison
 */

#ifndef UI_H_
#define UI_H_

typedef struct Data{
	Mapper* map;
	GridInfo* info;
	Matrix* matrix;
}Data;


void datafree(Data* data);
Data* datastep(char* filePath);
void trainstep(Data* data);
void wordtrain(Data* data);
void pyinf(Mapper* map, GridInfo* info);
void pcolinf(GridInfo* info);
void pfiles(char** fileNames, char* buf, size_t buflen, size_t* l, short* found);
void pdir(struct dirent* ent, char** fileNames, short* found);
void mtrprint(Matrix* matrix);
void mtrlmprint(Matrix* matrix, size_t rows);

#endif /* UI_H_ */
