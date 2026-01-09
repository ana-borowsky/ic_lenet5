#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"

void print_available_memory();
void resizeImage(uint8_t*, uint8_t*);
void preprocessImageData(uint8_t*, int, float*);
tflite::MicroInterpreter* initializeInterpreter(char*, const tflite::Model*, tflite::MicroMutableOpResolver<10>*, int, uint8_t*);
int predict(tflite::MicroInterpreter*, float*, int image_size);
void include_prediction_in_confusion_matrix(int*, char*, int);
void print_confusion_matrix(int*);
