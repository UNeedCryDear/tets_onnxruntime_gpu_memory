#pragma once
#pragma once
#include <iostream>
#include<memory>
#include <numeric>
#include<onnxruntime_cxx_api.h>

//#include <tensorrt_provider_factory.h>  //if use OrtTensorRTProviderOptionsV2
//#include <onnxruntime_c_api.h>

#define ORT_OLD_VISON 13

class TestOnnx {
public:
	TestOnnx() :_OrtMemoryInfo(Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtDeviceAllocator, OrtMemType::OrtMemTypeCPUOutput)) {};
	~TestOnnx() ;

public:

	bool ReadModel(const std::string& modelPath, bool isCuda = false, int cudaID = 0, bool warmUp = true);

private:

	template <typename T>
	T VectorProduct(const std::vector<T>& v)
	{
		return std::accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
	};

	const int _netWidth = 640;   //ONNX-net-input-width
	const int _netHeight = 640;  //ONNX-net-input-height

	int _batchSize = 4;  //if multi-batch,set this
	bool _isDynamicShape = false;//onnx support dynamic shape
	float _classThreshold = 0.5;
	float _nmsThreshold = 0.45;

	//ONNXRUNTIME	
	Ort::Env _OrtEnv = Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR, "Yolov5");
	Ort::SessionOptions _OrtSessionOptions = Ort::SessionOptions();
	Ort::Session* _OrtSession = nullptr;
	Ort::MemoryInfo _OrtMemoryInfo;
	Ort::AllocatorWithDefaultOptions _OrtAllocator;
	Ort::RunOptions _OrtRunOptions{ nullptr };
	OrtStatus* _OrtStatus{ nullptr };
#if ORT_API_VERSION < ORT_OLD_VISON

	char* _inputName, * _output_name0, * _output_name1;
#else
	std::shared_ptr<char> _inputName, _output_name0, _output_name1;
#endif
	std::vector<char*> _inputNodeNames; 
	std::vector<char*> _outputNodeNames;

	size_t _inputNodesNum = 0;      
	size_t _outputNodesNum = 0;       

	ONNXTensorElementDataType _inputNodeDataType;
	ONNXTensorElementDataType _outputNodeDataType;
	std::vector<int64_t> _inputTensorShape; 

	std::vector<int64_t> _outputTensorShape;
	std::vector<int64_t> _outputMaskTensorShape;
public:
	std::vector<std::string> _className = {
		"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
		"fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
		"elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
		"skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
		"tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
		"sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
		"potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
		"microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
		"hair drier", "toothbrush"
	};



};