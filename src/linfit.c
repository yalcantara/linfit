/*
 ============================================================================
 Name        : linfit.c
 Author      : Yaison Alcantara
 Version     : 1.0
 Description : Display the cost function of a linear model.
 ============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <math.h>
#include <errno.h>
#include <unistd.h>
#include <dirent.h>
#include <time.h>
#include <unistd.h>
#include <pthread.h>
#include <signal.h>

#include "utils.h"
#include "ml.h"
#include "ui.h"

int start();

/*
 * Reads any *.data file under the <CURRENT_DIR>/files directory, allowing the 
 * user to choose which dataset to work on. The program will parse the data
 * (asuming ',' is the delimiter), and build the X and y matrices, where X
 * is the features matrix. Next, compute the cost before and after
 * training and prints it in the standard output.
 */
int main(void) {

	//srand(time(0));
	int ret = start();
	return ret;
	//printf("%d", RAND_MAX);
}

/*
 * This is the equivalent main function of the program.
 */
int start() {

	srand((unsigned) time(NULL));

	char** fileNames = malloc(sizeof(char*) * 100);
	size_t buflen = 1000;
	short found = 0;
	//In buf, we are going to store the current working directoy path.
	char* buf = malloc(sizeof(char) * buflen);
	size_t l;
	
	
	pfiles(fileNames, buf, buflen, &l, &found);
	if (found == 0) {
		fprintf(stderr, "No data file found.\n");
		fflush(stderr);
		return EXIT_FAILURE;
	}

	//Here, we are going to display the file the user chose.
	printf("\nYour choose: ");
	flush();
	int val;
	scanf("%d", &val);
	val--;
	printf("\nSelected '%s'\n", fileNames[val]);

	strcpy(buf + l, fileNames[val]);

	for (size_t i = 0; i < found; i++) {
		free(fileNames[i]);
	}
	free(fileNames);

	//Now it's time to parse the dataset and build the Data struct.
	Data* data = datastep(buf);
	free(buf);
	printf("\n");
	if (data) {
		//If the datastep executed with no problem, the data pointer will
		//not be null. We can proceed with training.
		trainstep(data);

		datafree(data);
	}

	return EXIT_SUCCESS;
}
