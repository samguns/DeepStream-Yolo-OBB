#include <algorithm>
#include <iostream>
#include <vector>

#include <thrust/copy.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "nvdsinfer_custom_impl.h"

extern "C" bool
NvDsInferParseYoloOBBCuda(std::vector<NvDsInferLayerInfo> const& outputLayersInfo, NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams, std::vector<NvDsInferInstanceMaskInfo>& objectList);
extern "C" bool
NvDsInferParseYolo11OBBCuda(std::vector<NvDsInferLayerInfo> const& outputLayersInfo, NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams, std::vector<NvDsInferInstanceMaskInfo>& objectList);

static constexpr uint kObbPointCount = 4;
static constexpr uint kObbMaskElementCount = kObbPointCount * 2 + 1;
static constexpr float kNmsThreshold = 0.45f;

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

static std::vector<DecodedYoloOBBInfo> nonMaximumSuppression(std::vector<DecodedYoloOBBInfo> binfo)
{
  auto overlap1D = [](float x1min, float x1max, float x2min, float x2max) -> float {
    if (x1min > x2min) {
      std::swap(x1min, x2min);
      std::swap(x1max, x2max);
    }
    return x1max < x2min ? 0.0f : std::min(x1max, x2max) - x2min;
  };

  auto computeIoU = [&overlap1D](const DecodedYoloOBBInfo& bbox1, const DecodedYoloOBBInfo& bbox2) -> float {
    const NvDsInferInstanceMaskInfo& object1 = bbox1.object;
    const NvDsInferInstanceMaskInfo& object2 = bbox2.object;
    const float overlapX = overlap1D(object1.left, object1.left + object1.width, object2.left, object2.left + object2.width);
    const float overlapY = overlap1D(object1.top, object1.top + object1.height, object2.top, object2.top + object2.height);
    const float area1 = object1.width * object1.height;
    const float area2 = object2.width * object2.height;
    const float overlap2D = overlapX * overlapY;
    const float unionArea = area1 + area2 - overlap2D;
    return unionArea == 0.0f ? 0.0f : overlap2D / unionArea;
  };

  std::stable_sort(binfo.begin(), binfo.end(), [](const DecodedYoloOBBInfo& b1, const DecodedYoloOBBInfo& b2) {
    return b1.object.detectionConfidence > b2.object.detectionConfidence;
  });

  std::vector<DecodedYoloOBBInfo> out;
  out.reserve(binfo.size());
  for (const DecodedYoloOBBInfo& candidate : binfo) {
    bool keep = true;
    for (const DecodedYoloOBBInfo& selected : out) {
      if (candidate.object.classId == selected.object.classId && computeIoU(candidate, selected) > kNmsThreshold) {
        keep = false;
        break;
      }
    }
    if (keep) {
      out.push_back(candidate);
    }
  }

  return out;
}

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

static bool NvDsInferParseCustomYolo11OBBCuda(std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
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
  std::vector<DecodedYoloOBBInfo> nmsDecoded(hostDecoded.begin(), hostDecoded.end());
  nmsDecoded = nonMaximumSuppression(nmsDecoded);

  objectList.clear();
  objectList.reserve(nmsDecoded.size());

  for (const DecodedYoloOBBInfo& decodedObject : nmsDecoded) {
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

extern "C" bool
NvDsInferParseYolo11OBBCuda(std::vector<NvDsInferLayerInfo> const& outputLayersInfo, NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams, std::vector<NvDsInferInstanceMaskInfo>& objectList)
{
  return NvDsInferParseCustomYolo11OBBCuda(outputLayersInfo, networkInfo, detectionParams, objectList);
}

CHECK_CUSTOM_INSTANCE_MASK_PARSE_FUNC_PROTOTYPE(NvDsInferParseYoloOBBCuda);
CHECK_CUSTOM_INSTANCE_MASK_PARSE_FUNC_PROTOTYPE(NvDsInferParseYolo11OBBCuda);