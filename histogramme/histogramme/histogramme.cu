#include <iostream>
#include <fstream>
#include <string>

using namespace std;

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <ctime>


void mainHistogramme (void) {
	unsigned char histo[256];
	for (int i = 0; i < 256; i++)
	{
		histo[i] = 0;
	}

	clock_t t;
	t = clock();

	ifstream fichier("document.txt", ios::in);
    if(fichier){
        char caractere;
        while(fichier.get(caractere)){
			histo[(unsigned char) caractere]++;
        }
        fichier.close();
    }else
        cerr << "Impossible d'ouvrir le fichier !" << endl;
	
	t = clock() - t;
	cout<< "Temps d'exec : "<<(((float)t)/CLOCKS_PER_SEC) * 1000 << " ms" << endl;

	unsigned int total = 0;
	for (int i = 0; i < 256; i++)
	{
		total += histo[i];
	}
	cout << total << endl;
	
	system("PAUSE");
}