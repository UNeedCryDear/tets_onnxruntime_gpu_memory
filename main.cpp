#include <iostream>
#include<math.h>
#include "test_gpu_memory.h"
#include<time.h>

using namespace std;


int yolov5_seg_onnx()
{

	string model_path = "./models/yolov5n.onnx";
	TestOnnx* test_api = new TestOnnx();
	if (test_api->ReadModel(model_path, true, 0, true)) {
		cout << "read net ok!" << endl;
	}
	else {
		return -1;
	}

	delete test_api;
	test_api = nullptr;
	return 0;
}

int main() {

	yolov5_seg_onnx(); //OnnxRuntime,support dynamic!
	return 0;
}


