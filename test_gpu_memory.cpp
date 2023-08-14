#include "test_gpu_memory.h"
using namespace std;
using namespace Ort;

TestOnnx::~TestOnnx() {

	Ort::detail::OrtRelease(_OrtStatus);
	Ort::detail::OrtRelease(_OrtRunOptions);
	Ort::detail::OrtRelease(*_OrtSession);

	//Error: 
   //Ort::detail::OrtRelease(_OrtEnv);
   //Ort::detail::OrtRelease(_OrtSessionOptions);
   //Ort::detail::OrtRelease(_OrtAllocator);
   //Ort::detail::OrtRelease(_OrtMemoryInfo);   

};


bool TestOnnx::ReadModel(const std::string& modelPath, bool isCuda, int cudaID, bool warmUp) {
	if (_batchSize < 1) _batchSize = 1;
	try
	{
		std::vector<std::string> available_providers = GetAvailableProviders();
		auto cuda_available = std::find(available_providers.begin(), available_providers.end(), "CUDAExecutionProvider");
		if (isCuda && (cuda_available == available_providers.end()))
		{
			std::cout << "Your ORT build without GPU. Change to CPU." << std::endl;
			std::cout << "************* Infer model on CPU! *************" << std::endl;
		}
		else if (isCuda && (cuda_available != available_providers.end()))
		{
#if ORT_API_VERSION < ORT_OLD_VISON
			OrtCUDAProviderOptions cudaOption;
			cudaOption.device_id = cudaID;
			_OrtSessionOptions.AppendExecutionProvider_CUDA(cudaOption);
#else
			_OrtStatus = OrtSessionOptionsAppendExecutionProvider_CUDA(_OrtSessionOptions, cudaID);
#endif
		}
		else
		{
			std::cout << "************* Infer model on CPU! *************" << std::endl;
		}

		_OrtSessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

#ifdef _WIN32
		std::wstring model_path(modelPath.begin(), modelPath.end());
		_OrtSession = new Ort::Session(_OrtEnv, model_path.c_str(), _OrtSessionOptions);
#else
		_OrtSession = new Ort::Session(_OrtEnv, modelPath.c_str(), _OrtSessionOptions);
#endif


		//init input
		_inputNodesNum = _OrtSession->GetInputCount();

#if ORT_API_VERSION < ORT_OLD_VISON
		_inputName = _OrtSession->GetInputName(0, _OrtAllocator);
		_inputNodeNames.push_back(_inputName);
#else
		_inputName = std::move(_OrtSession->GetInputNameAllocated(0, _OrtAllocator));
		_inputNodeNames.push_back(_inputName.get());
#endif

		Ort::TypeInfo inputTypeInfo = _OrtSession->GetInputTypeInfo(0);
		auto input_tensor_info = inputTypeInfo.GetTensorTypeAndShapeInfo();
		_inputNodeDataType = input_tensor_info.GetElementType();
		_inputTensorShape = input_tensor_info.GetShape();

		if (_inputTensorShape[0] == -1)
		{
			_isDynamicShape = true;
			_inputTensorShape[0] = _batchSize;

		}
		if (_inputTensorShape[2] == -1 || _inputTensorShape[3] == -1) {
			_isDynamicShape = true;
			_inputTensorShape[2] = _netHeight;
			_inputTensorShape[3] = _netWidth;
		}
		//init output
		_outputNodesNum = _OrtSession->GetOutputCount();
#if ORT_API_VERSION < ORT_OLD_VISON
		_output_name0 = _OrtSession->GetOutputName(0, _OrtAllocator);

#else
		_output_name0 = std::move(_OrtSession->GetOutputNameAllocated(0, _OrtAllocator));

#endif
		Ort::TypeInfo type_info_output0(nullptr);
		Ort::TypeInfo type_info_output1(nullptr);
		bool flag = false;
		type_info_output0 = _OrtSession->GetOutputTypeInfo(0);  //output0
#if ORT_API_VERSION < ORT_OLD_VISON
		_outputNodeNames.push_back(_output_name0);

#else
		_outputNodeNames.push_back(_output_name0.get());

#endif

		auto tensor_info_output0 = type_info_output0.GetTensorTypeAndShapeInfo();
		_outputNodeDataType = tensor_info_output0.GetElementType();
		_outputTensorShape = tensor_info_output0.GetShape();

		//warm up
		if (isCuda && warmUp) {
			//draw run
			cout << "Start warming up" << endl;
			size_t input_tensor_length = VectorProduct(_inputTensorShape);
			float* temp = new float[input_tensor_length];
			std::vector<Ort::Value> input_tensors;
			std::vector<Ort::Value> output_tensors;
			input_tensors.push_back(Ort::Value::CreateTensor<float>(
				_OrtMemoryInfo, temp, input_tensor_length, _inputTensorShape.data(),
				_inputTensorShape.size()));
			for (int i = 0; i < 3; ++i) {
				output_tensors = _OrtSession->Run(_OrtRunOptions,
					_inputNodeNames.data(),
					input_tensors.data(),
					_inputNodeNames.size(),
					_outputNodeNames.data(),
					_outputNodeNames.size());
			}

			delete[]temp;
		}
	}
	catch (const std::exception&) {
		return false;
	}

}

