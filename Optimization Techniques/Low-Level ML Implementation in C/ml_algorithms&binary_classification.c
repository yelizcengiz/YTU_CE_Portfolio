#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define N 50
#define VECTOR_SIZE (N * N + 1)
#define VECTOR_CHAR_SIZE 50000
#define TRAINING_SET_SIZE 2000
#define MAX_ITERATION 500
#define EXPECTED_ERROR 0.001
#define LEARNING_RATE 0.0001

#define DEV 1

typedef struct {
    double data[VECTOR_SIZE];
    double type; 
} Vector;

double matrisCarpim (double*, Vector*);
double tanhTurev (double);
void initializeWeight(double*,double);
double gradientDescent(Vector*,double*,int, FILE*);
double stochasticGradientDescent(Vector* , double* , int ,FILE* );
double adamOptimization(Vector* , double* , int ,FILE* );

void shuffleVectors(Vector* , int );
double testModel(Vector* , int , double* );
void writeResultToCSV(FILE* file, int epoch, double loss, double time) {
    fprintf(file, "%d,%.6f,%.6f\n", epoch, loss, time);
}


int main() {
	double firstWeight[5]={-0.003,-0.002, 0.00, 0.002, 0.003};
	
	int vector_count = 0;
    int testSize ;
    int trainSize;
	char filename[30];
	int i;
    FILE *file = fopen("imagedata.csv", "r");
    if (!file) {
        printf("Dosya acilamadi\n");
        return 1;
    }
	
	Vector *vectors = malloc(TRAINING_SET_SIZE * sizeof(Vector));
	if (!vectors) {
		printf("Bellek tahsisi basarisiz.\n");
		fclose(file);
		return 1;
	}
	       
	char *line = malloc(VECTOR_CHAR_SIZE * sizeof(char));
	if (!line) {
		printf("Line icin bellek tahsisi basarisiz.\n");
		fclose(file);
		free(vectors);
		return 1;
	}
	
	double* weights = malloc(VECTOR_SIZE * sizeof(double));
    if (!weights) {
        printf("\nAgirlik malloc hata.\n");
        free(vectors);
        free(line);
        fclose(file);
        return 1;
    }
	
    while (fgets(line, VECTOR_CHAR_SIZE, file) && vector_count < TRAINING_SET_SIZE && !feof(file)) {
        char *token = strtok(line, ",");
        i = 0;

        while (token != NULL) {
            if (i < VECTOR_SIZE-1) {
                vectors[vector_count].data[i] = atof(token);
            }
			else {
            	vectors[vector_count].data[i] = atof(token); //bias
            	vectors[vector_count].type = atof(token);	 //type
            }
            token = strtok(NULL, ",");
            i++;
        }
        vector_count++;
    }
    free(line);
	fclose(file);
	printf("\nToplam %d vektor okundu\n", vector_count);
    
    shuffleVectors(vectors, vector_count);
    
    testSize = vector_count * 0.2;  // Test(%20)
    trainSize = vector_count - testSize;
	
	printf("Egitim seti: %d vektor\n", trainSize);
	printf("Test seti: %d vektor\n", testSize);
	
    Vector* testSet = malloc(testSize * sizeof(Vector));
    Vector* trainSet = malloc(trainSize * sizeof(Vector));

    if (!testSet || !trainSet) {
        free(vectors);
        free(weights);
        return 1;
    }
	
    for (i = 0; i < testSize; i++) {
        testSet[i] = vectors[i];
    }

    for (i = 0; i < trainSize; i++) {
        trainSet[i] = vectors[testSize + i];
    }
	
	printf("\n\n- GRADIENT DESCENT -\n");
    for(i = 0; i < 5; i++){
    	sprintf(filename, "Gradient_D_W%d.csv", i + 1);
    	FILE* gdFile = fopen(filename, "w");
    	fprintf(gdFile, "epoch,loss,time\n");
    	
    	printf("\nBaslangic agirligi: %.4f ile egitim basliyor...\n", firstWeight[i]);
		initializeWeight(weights, firstWeight[i]);
    	double finalLoss = gradientDescent(trainSet, weights, trainSize, gdFile);
    	printf("Egitim tamamlandi. Final Loss: %.6f\n", finalLoss);
    	fclose(gdFile);
    	
    	printf("\nTest\n");
    	testModel(testSet, testSize, weights);
	}
    
    
    printf("\n\n- STOCHASTIC GRADIENT DESCENT -\n");
    for(i = 0; i < 5; i++){
    	sprintf(filename, "S_Gradient_D_W%d.csv", i + 1);
    	FILE* sgdFile = fopen(filename, "w");
    	fprintf(sgdFile, "epoch,loss,time\n");
    	
    	printf("\nBaslangic agirligi: %.4f ile egitim basliyor...\n", firstWeight[i]);
		initializeWeight(weights, firstWeight[i]);
    	double finalLoss = stochasticGradientDescent(trainSet, weights, trainSize, sgdFile);
    	printf("Egitim tamamlandi. Final Loss: %.6f\n", finalLoss);
    	fclose(sgdFile);
    	
    	printf("\nTest\n");
    	testModel(testSet, testSize, weights);
	}
    
    
    printf("\n\n- ADAM OPTIMIZATION -\n");
    for(i = 0; i < 5; i++){
    	sprintf(filename, "ADAM_W%d.csv", i + 1);
    	FILE* adamFile = fopen(filename, "w");
    	fprintf(adamFile, "epoch,loss,time\n");
    	
    	printf("\nBaslangic agirligi: %.4f ile egitim basliyor...\n", firstWeight[i]);
		initializeWeight(weights, firstWeight[i]);
    	double finalLoss = adamOptimization(trainSet, weights, trainSize, adamFile);
    	printf("Egitim tamamlandi. Final Loss: %.6f\n", finalLoss);
    	fclose(adamFile);
    	
    	printf("\nTest\n");
    	testModel(testSet, testSize, weights);
	}
    
    printf("Program Tamamlandi\n");
    
    free(testSet);
    free(trainSet);
    free(weights);
    free(vectors);
	
    return 0;
}



void shuffleVectors(Vector* vectors, int size) {
    srand(time(NULL));
    int i, j;
    for (i = size - 1; i > 0; i--) {
        j = rand() % (i + 1);
        Vector temp = vectors[i];
        vectors[i] = vectors[j];
        vectors[j] = temp;
    }
}

double matrisCarpim (double* w, Vector* v){
	double result = 0.0;
	int i;
	for(i = 0; i < VECTOR_SIZE; i++){
		result += w[i] * v->data[i];
	}
	return result;
}

double tanhTurev (double a){
	return 1.0 - tanh(a) * tanh(a); 
}

void initializeWeight(double* weight, double value){
	int i;
	for(i = 0 ; i < VECTOR_SIZE ; i++){
		weight[i] = value;
	}
}

double gradientDescent(Vector* vec, double* weights, int setSize, FILE* file){
	clock_t start = clock();
	double loss = 1, error, totalError = 0;
	double temp, predictedY, gradient;
	int i, j, step = 0;
	double zaman = 0;
		
	while(loss > EXPECTED_ERROR && step < MAX_ITERATION)
	{
		totalError = 0;
		for(i = 0; i < setSize; i++){
			temp = matrisCarpim(weights, &vec[i]);
			predictedY = tanh(temp);
			error = predictedY - vec[i].type; 
			
			totalError += (error * error) / 2;
			
			gradient = error * tanhTurev(temp);
			
			for(j = 0; j < VECTOR_SIZE; j++){
				weights[j] -= LEARNING_RATE * gradient * vec[i].data[j];
			}		
		}
		loss = totalError / setSize;  
		zaman = (double)(clock() - start) / CLOCKS_PER_SEC;
		writeResultToCSV(file, step, loss, zaman);
	    step++;
	    
	    if(step % 100 == 0 && DEV == 1)
	    	printf("  Step: %4d | Loss: %.6f | Zaman: %.2f sn\n", step, loss, zaman);
	}
	
	printf("  Toplam iterasyon: %d\n", step);
	return loss;
}

double stochasticGradientDescent(Vector* vec, double* weights, int setSize, FILE* file){
	clock_t start = clock();
	double loss = 1, error, totalError = 0;
	double temp, predictedY, gradient;
	int miniBatch = 10;
	int i, j, k, step = 0;
	float zaman = 0, eps = LEARNING_RATE;
	
	while(loss > EXPECTED_ERROR && step < MAX_ITERATION)
	{
		totalError = 0;
		for(i = 0; i < miniBatch; i++){ 
			k = rand() % setSize;
			temp = matrisCarpim(weights, &vec[k]);
			predictedY = tanh(temp);
			error = predictedY - vec[k].type;
			
			totalError += (error * error) / 2;
			
			gradient = error * tanhTurev(temp);
			
			for(j = 0; j < VECTOR_SIZE; j++){
				weights[j] -= eps * gradient * vec[k].data[j];
			}		
		}
		loss = totalError / miniBatch;
	    step++;
	    zaman = (double)(clock() - start) / CLOCKS_PER_SEC;
	    writeResultToCSV(file, step, loss, zaman);
	    
	    if (fmod(zaman, 0.005f) == 0.0f) {
		    eps *= 0.9f;
		}
	    
		if(step % 100 == 0 && DEV == 1)
	    	printf("  Step: %4d | Loss: %.6f | Zaman: %.2f sn | LR: %.6f\n", 
	    	       step, loss, zaman, eps);	
	}
	
	printf("  Toplam iterasyon: %d\n", step);
	return loss;
}

double adamOptimization(Vector* vec, double* weights, int setSize, FILE* file) {
    clock_t start = clock();  
    double beta1 = 0.9;     
    double beta2 = 0.999;   
    double epsilon = 1e-8; 
    double loss = 1, error, totalError = 0;
    double temp, predictedY, gradient;
    int i, j, step = 0;
    double zaman = 0;
    double m_hat, v_hat;
    
    double* m = malloc(VECTOR_SIZE * sizeof(double));
    double* v = malloc(VECTOR_SIZE * sizeof(double));
    
    if(!m || !v) {
        printf("ADAM memory allocation error\n");
        return -1;
    }
    
    for(i = 0; i < VECTOR_SIZE; i++) {
        m[i] = 0.0;
        v[i] = 0.0;
    }
    
    while(loss > EXPECTED_ERROR && step < MAX_ITERATION) {
        totalError = 0;
        for(i = 0; i < setSize; i++) {
            temp = matrisCarpim(weights, &vec[i]);
            predictedY = tanh(temp);
            error = predictedY - vec[i].type;          
            totalError += (error * error) / 2;
            gradient = error * tanhTurev(temp);
            
            for(j = 0; j < VECTOR_SIZE; j++) {
                m[j] = beta1 * m[j] + (1 - beta1) * gradient * vec[i].data[j];
                v[j] = beta2 * v[j] + (1 - beta2) * pow(gradient * vec[i].data[j], 2);
                m_hat = m[j] / (1 - pow(beta1, step + 1)); 
                v_hat = v[j] / (1 - pow(beta2, step + 1)); 
                
                weights[j] -= LEARNING_RATE * m_hat / (sqrt(v_hat) + epsilon);
            }
        }
        loss = totalError / setSize;
        step++;
        zaman = (double)(clock() - start) / CLOCKS_PER_SEC;
        writeResultToCSV(file, step, loss, zaman);
        
        if(step % 100 == 0 && DEV == 1)
            printf("  Step: %4d | Loss: %.6f | Zaman: %.2f sn\n", step, loss, zaman);
    }
    
    printf("  Toplam iterasyon: %d\n", step);
    free(m);
    free(v);  
    return loss;
}

double testModel(Vector* testSet, int testSize, double* weights) {
    double totalError = 0.0;
    int correct = 0, i;
    
    for (i = 0; i < testSize; i++) {
        double temp = matrisCarpim(weights, &testSet[i]);
        double predictedY = tanh(temp);
        double actualY = testSet[i].type;
        
        double error = predictedY - actualY;
        totalError += (error * error) / 2;

        if(predictedY >= 0 && testSet[i].type == 1){
        	correct++;
		}
		else if(predictedY < 0 && testSet[i].type == -1){
        	correct++;
		}
    }
    
    double accuracy = (double)correct / testSize * 100.0;
    double avgError = totalError / testSize;
    
    printf("  ------------------------\n");
    printf("  Test Sonuclari:\n");
    printf("  Dogruluk (Accuracy): %.2f%%\n", accuracy);
    printf("  Dogru tahmin: %d / %d\n", correct, testSize);
    printf("  Ortalama hata: %.6f\n", avgError);
    
    return avgError;
}
