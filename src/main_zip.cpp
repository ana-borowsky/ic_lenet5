#include <Arduino.h>
#include <SPIFFS.h>
#include <esp_timer.h>

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/core/c/c_api_types.h"
#include "miniz.h"

#include "ml.h"

SET_LOOP_TASK_STACK_SIZE(64 * 1024);

namespace {
  const tflite::Model* model = nullptr;
  tflite::MicroMutableOpResolver<10>* resolver = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* input = nullptr;
  TfLiteTensor* output = nullptr;

  int tensor_arena_size = 256 * 1024;
  uint8_t* tensor_arena = nullptr;
}

void setup_z() {
  delay(3000);
  Serial.begin(115200);
  SPIFFS.begin();
  print_available_memory();

  char model_file[31] = "/spiffs/mobilenet.tflite";//mobilenet
  interpreter = initializeInterpreter(model_file, model, resolver, tensor_arena_size, tensor_arena);

  print_available_memory();
}

void loop_z() {
  char zip_file[31] = "/spiffs/test.zip";
  mz_bool status;
  mz_zip_archive zip_archive;

  memset(&zip_archive, 0, sizeof(zip_archive));

  status = mz_zip_reader_init_file(&zip_archive, zip_file, 0);
  if (!status) {
    printf("Zip file init failed!\n");
    return;
  }

  print_available_memory();

  mz_uint numFiles = mz_zip_reader_get_num_files(&zip_archive);

  //int lenet_image_size = 28 * 28 * 1;
  int mobilenet_image_size = 32 * 32 * 1;//mobilenet
  size_t image_size = mobilenet_image_size;//mobilenet
  //size_t image_size = lenet_image_size;
  // uint8_t* image_data;
  //uint8_t* image_data = (uint8_t*)malloc(lenet_image_size);
  //float* float_image_data = (float*)malloc(lenet_image_size * 4);
  uint8_t *image_data = (uint8_t *)malloc(mobilenet_image_size);
  float *float_image_data = (float *)malloc(mobilenet_image_size * sizeof(float));
  int prediction;
  int predictions[100] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  int64_t start = esp_timer_get_time();

  for (mz_uint i = 0; i < numFiles; i++) {
    mz_zip_archive_file_stat file_stat;
    if (!mz_zip_reader_file_stat(&zip_archive, i, &file_stat)) {
       printf("Zip file stat failed!\n");
       mz_zip_reader_end(&zip_archive);
       break;
    }

    // printf("Filename: \"%s\", Comment: \"%s\", Uncompressed size: %u, Compressed size: %u, Is Dir: %u\n", file_stat.m_filename, file_stat.m_comment, (uint)file_stat.m_uncomp_size, (uint)file_stat.m_comp_size, mz_zip_reader_is_file_a_directory(&zip_archive, i));
    printf("%d\n", i);

    if (!mz_zip_reader_is_file_a_directory(&zip_archive, i)) {

      // size_t* extracted_size;
      // image_data = (uint8_t*)mz_zip_reader_extract_to_heap(&zip_archive, i, extracted_size, 0);
      // if (!extracted_size == lenet_image_size) {
      
      status = mz_zip_reader_extract_to_mem(&zip_archive, i, image_data, image_size, 0);
      if (!status) {
        printf("Zip file extraction failed!\n");
        mz_zip_reader_end(&zip_archive);
        break;
      }

      //ocessImageData(image_data, lenet_image_size, float_image_data);
      preprocessImageData(image_data, mobilenet_image_size, float_image_data);
      //prediction = predict(interpreter, float_image_data, lenet_image_size); //lenet
      prediction = predict(interpreter, float_image_data, mobilenet_image_size); // mobilenet
      include_prediction_in_confusion_matrix(predictions, file_stat.m_filename, prediction);
    }
  }

  int64_t stop = esp_timer_get_time();
  printf("Elapsed time: %lld microseconds\n", stop - start);

  print_confusion_matrix(predictions);

  print_available_memory();

  free(image_data);
  free(float_image_data);

  mz_zip_reader_end(&zip_archive);
  
  delay(60000);
}