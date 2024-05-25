#include <TensorFlowLite.h>
#include "model_int.h"             // Include the model header
#include "images.h"             // Include the image header
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include <Arduino.h>

const tflite::Model* tflModel = nullptr;
tflite::MicroInterpreter* tflInterpreter = nullptr;
TfLiteTensor* tflInputTensor = nullptr;
TfLiteTensor* tflOutputTensor = nullptr;
tflite::AllOpsResolver tflOpsResolver;
constexpr int tensorArenaSize = 50 * 1024;
byte tensorArena[tensorArenaSize] __attribute__((aligned(16)));

// Array of image pointers and their sizes
const int* images[] = {benchmark_image, mnist_sample_0, mnist_sample_1, mnist_sample_2, mnist_sample_3, mnist_sample_4};
const int image_sizes[] = {benchmark_image_size, mnist_sample_size_0, mnist_sample_size_1, mnist_sample_size_2, mnist_sample_size_3, mnist_sample_size_4};
const char* image_names[] = {"benchmark_image", "mnist_sample_0", "mnist_sample_1", "mnist_sample_2", "mnist_sample_3", "mnist_sample_4"};
constexpr int num_images = 6;  // Total number of images

void setup() {
    Serial.begin(9600);
    tflModel = tflite::GetModel(full_int_quantized_model_tflite);
    if (tflModel->version() != TFLITE_SCHEMA_VERSION) {
        Serial.println("Model schema mismatch!");
        while (1); 
    }
    tflInterpreter = new tflite::MicroInterpreter(tflModel, tflOpsResolver, tensorArena, tensorArenaSize, nullptr);
    if (tflInterpreter->AllocateTensors() != kTfLiteOk) {
        Serial.println("Failed to allocate tensors!");
        return;
    }
    tflInputTensor = tflInterpreter->input(0);
    tflOutputTensor = tflInterpreter->output(0);
    Serial.println("Model and interpreter ready!");
}

void normalizeImage(uint8_t* buffer, const int* image, int len) {
    for (int i = 0; i < len; i++) {
        buffer[i] = static_cast<uint8_t>(image[i] / 255.0f * 255); // Normalize and scale to 0-255
    }
}

void loop() {
    for (int i = 0; i < num_images; i++) {
        // Prepare the input tensor
        tflInputTensor = tflInterpreter->input(0);
        normalizeImage(tflInputTensor->data.uint8, images[i], image_sizes[i]);

        // Run inference
        if (tflInterpreter->Invoke() != kTfLiteOk) {
            Serial.println("Failed to invoke interpreter!");
            return;
        }

        // Process and print the output
        tflOutputTensor = tflInterpreter->output(0);
        float scale = tflOutputTensor->params.scale;
        int zero_point = tflOutputTensor->params.zero_point;

        uint8_t* probabilities = tflOutputTensor->data.uint8;
        int numClasses = tflOutputTensor->dims->data[1];
        
        Serial.print("Image: ");
        Serial.print(image_names[i]);
        Serial.println(" Class probabilities:");
        for (int j = 0; j < numClasses; j++) {
            float probability = (probabilities[j] - zero_point) * scale;
            Serial.print("Class ");
            Serial.print(j);
            Serial.print(": ");
            Serial.println(probability, 6);  // Print probabilities with precision
        }

        delay(1000);  // Delay before processing the next image
    }
    // while (1);  // Stop the loop afte+r one run through all images
}
