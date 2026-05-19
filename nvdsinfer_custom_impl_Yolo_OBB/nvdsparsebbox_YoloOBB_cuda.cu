#include <algorithm>
#include <iostream>

#include <thrust/copy.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "nvdsinfer_custom_impl.h"

extern "C" bool
NvDsInferParseYoloOBBCuda(std::vector<NvDsInferLayerInfo> const& outputLayersInfo, NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams, std::vector<NvDsInferInstanceMaskInfo>& objectList);

static constexpr uint kObbPointCount = 4;
static constexpr uint kObbMaskElementCount = kObbPointCount * 2 + 1;

struct DecodedYoloOBBInfo
{
  NvDsInferInstanceMaskInfo object;
  float corners[kObbPointCount * 2];
  float angle;
};

struct IsValidYoloOBBInfo
{
  __host__ __device__ bool operator()(const DecodedYoloOBBInfo& decoded) const
  {
    return decoded.object.detectionConfidence > 0.0f && decoded.object.width >= 1.0f && decoded.object.height >= 1.0f;
  }
};

__device__ float clampFloat(const float value, const float minValue, const float maxValue)
{
  return fminf(maxValue, fmaxf(minValue, value));
}

__global__ void decodeTensorYoloOBBCuda(DecodedYoloOBBInfo *decoded, const float* output_tensor,
    const uint outputSize, const uint netW, const uint netH,
    const float minPreclusterThreshold)
{
  int x_id = blockIdx.x * blockDim.x + threadIdx.x;

  if (x_id >= outputSize) {
    return;
  }

  float maxProb = output_tensor[x_id * 7 + 4];
  int maxIndex = (int) output_tensor[x_id * 7 + 5];

  if (maxProb < minPreclusterThreshold) {
    decoded[x_id].object.detectionConfidence = 0.0;
    return;
  }

  float bxc = output_tensor[x_id * 7 + 0];
  float byc = output_tensor[x_id * 7 + 1];
  float bw = output_tensor[x_id * 7 + 2];
  float bh = output_tensor[x_id * 7 + 3];
  float angle = output_tensor[x_id * 7 + 6];

  decoded[x_id].angle = angle;
  NvDsInferInstanceMaskInfo& binfo = decoded[x_id].object;
  binfo.classId = 0;
  binfo.left = 0.0f;
  binfo.top = 0.0f;
  binfo.width = 0.0f;
  binfo.height = 0.0f;
  binfo.detectionConfidence = 0.0f;
  binfo.mask = nullptr;

  float halfW = bw * 0.5f;
  float halfH = bh * 0.5f;
  float cosA = cosf(angle);
  float sinA = sinf(angle);

  float localX[kObbPointCount] = {-halfW, halfW, halfW, -halfW};
  float localY[kObbPointCount] = {-halfH, -halfH, halfH, halfH};

  for (uint point = 0; point < kObbPointCount; ++point) {
    float cornerX = bxc + localX[point] * cosA - localY[point] * sinA;
    float cornerY = byc + localX[point] * sinA + localY[point] * cosA;

    cornerX = clampFloat(cornerX, 0.0f, float(netW));
    cornerY = clampFloat(cornerY, 0.0f, float(netH));

    decoded[x_id].corners[point * 2 + 0] = cornerX;
    decoded[x_id].corners[point * 2 + 1] = cornerY;
  }

  binfo.left = clampFloat(bxc - halfW, 0.0f, float(netW));
  binfo.top = clampFloat(byc - halfH, 0.0f, float(netH));
  binfo.width = fminf(float(netW), fmaxf(float(0.0), bxc + halfW - binfo.left));
  binfo.height = fminf(float(netH), fmaxf(float(0.0), byc + halfH - binfo.top));
  binfo.detectionConfidence = maxProb;
  binfo.classId = maxIndex;
}

static bool NvDsInferParseCustomYoloOBBCuda(std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo, NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferInstanceMaskInfo>& objectList)
{
    if (outputLayersInfo.empty()) {
        std::cerr << "ERROR: Could not find output layer in bbox parsing" << std::endl;
        return false;
    }

    const NvDsInferLayerInfo& output_tensor = outputLayersInfo[0];
    const int outputSize = output_tensor.inferDims.d[0];

    thrust::device_vector<DecodedYoloOBBInfo> decoded(outputSize);

    float minPreclusterThreshold = *(std::min_element(detectionParams.perClassPreclusterThreshold.begin(),
        detectionParams.perClassPreclusterThreshold.end()));

    int threads_per_block = 1024;
    int number_of_blocks = ((outputSize - 1) / threads_per_block) + 1;

    decodeTensorYoloOBBCuda<<<number_of_blocks, threads_per_block>>>(
        thrust::raw_pointer_cast(decoded.data()), (float*) (output_tensor.buffer), outputSize, networkInfo.width,
        networkInfo.height, minPreclusterThreshold);

    thrust::device_vector<DecodedYoloOBBInfo> validDecoded(outputSize);
    auto validEnd = thrust::copy_if(decoded.begin(), decoded.end(), validDecoded.begin(), IsValidYoloOBBInfo());
    validDecoded.resize(validEnd - validDecoded.begin());

    thrust::host_vector<DecodedYoloOBBInfo> hostDecoded = validDecoded;

    objectList.clear();
    objectList.reserve(hostDecoded.size());

    for (const DecodedYoloOBBInfo& decodedObject : hostDecoded) {
        NvDsInferInstanceMaskInfo object = decodedObject.object;
        object.mask = new float[kObbMaskElementCount];
        for (uint point = 0; point < kObbPointCount; ++point) {
            object.mask[point * 2 + 0] = decodedObject.corners[point * 2 + 0];
            object.mask[point * 2 + 1] = decodedObject.corners[point * 2 + 1];
        }
        object.mask[kObbPointCount * 2] = decodedObject.angle;
        object.mask_width = networkInfo.width;
        object.mask_height = networkInfo.height;
        object.mask_size = sizeof(float) * kObbMaskElementCount;

        objectList.push_back(object);
    }

    return true;
}

extern "C" bool
NvDsInferParseYoloOBBCuda(std::vector<NvDsInferLayerInfo> const& outputLayersInfo, NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams, std::vector<NvDsInferInstanceMaskInfo>& objectList)
{
  return NvDsInferParseCustomYoloOBBCuda(outputLayersInfo, networkInfo, detectionParams, objectList);
}

CHECK_CUSTOM_INSTANCE_MASK_PARSE_FUNC_PROTOTYPE(NvDsInferParseYoloOBBCuda);
