#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

#include "nvdsinfer_custom_impl.h"

extern "C" bool
NvDsInferParseYoloOBB(std::vector<NvDsInferLayerInfo> const& outputLayersInfo, NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams, std::vector<NvDsInferInstanceMaskInfo>& objectList);
extern "C" bool
NvDsInferParseYolo11OBB(std::vector<NvDsInferLayerInfo> const& outputLayersInfo, NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams, std::vector<NvDsInferInstanceMaskInfo>& objectList);

static constexpr unsigned int kObbPointCount = 4;
static constexpr unsigned int kObbMaskElementCount = kObbPointCount * 2 + 1;
static constexpr float kNmsThreshold = 0.45f;

struct DecodedYoloOBBInfo
{
  NvDsInferInstanceMaskInfo object;
  float corners[kObbPointCount * 2];
  float angle;
};

static float clampFloat(const float value, const float minValue, const float maxValue)
{
  return std::min(maxValue, std::max(minValue, value));
}

static bool isValidYoloOBBInfo(const DecodedYoloOBBInfo& decoded)
{
  return decoded.object.detectionConfidence > 0.0f && decoded.object.width >= 1.0f && decoded.object.height >= 1.0f;
}

static float overlap1D(float x1min, float x1max, float x2min, float x2max)
{
  if (x1min > x2min) {
    std::swap(x1min, x2min);
    std::swap(x1max, x2max);
  }
  return x1max < x2min ? 0.0f : std::min(x1max, x2max) - x2min;
}

static float computeIoU(const DecodedYoloOBBInfo& bbox1, const DecodedYoloOBBInfo& bbox2)
{
  const NvDsInferInstanceMaskInfo& object1 = bbox1.object;
  const NvDsInferInstanceMaskInfo& object2 = bbox2.object;
  const float overlapX = overlap1D(object1.left, object1.left + object1.width, object2.left, object2.left + object2.width);
  const float overlapY = overlap1D(object1.top, object1.top + object1.height, object2.top, object2.top + object2.height);
  const float area1 = object1.width * object1.height;
  const float area2 = object2.width * object2.height;
  const float overlap2D = overlapX * overlapY;
  const float unionArea = area1 + area2 - overlap2D;
  return unionArea <= 0.0f ? 0.0f : overlap2D / unionArea;
}

static std::vector<DecodedYoloOBBInfo> nonMaximumSuppression(std::vector<DecodedYoloOBBInfo> detections)
{
  std::stable_sort(detections.begin(), detections.end(), [](const DecodedYoloOBBInfo& b1, const DecodedYoloOBBInfo& b2) {
    return b1.object.detectionConfidence > b2.object.detectionConfidence;
  });

  std::vector<DecodedYoloOBBInfo> selected;
  selected.reserve(detections.size());
  for (const DecodedYoloOBBInfo& candidate : detections) {
    bool keep = true;
    for (const DecodedYoloOBBInfo& existing : selected) {
      if (candidate.object.classId == existing.object.classId && computeIoU(candidate, existing) > kNmsThreshold) {
        keep = false;
        break;
      }
    }
    if (keep) {
      selected.push_back(candidate);
    }
  }

  return selected;
}

static DecodedYoloOBBInfo decodeTensorYoloOBB(const float* outputTensor, const unsigned int rowIndex,
    const unsigned int netW, const unsigned int netH, const float minPreclusterThreshold)
{
  DecodedYoloOBBInfo decoded{};
  NvDsInferInstanceMaskInfo& binfo = decoded.object;
  binfo.classId = 0;
  binfo.left = 0.0f;
  binfo.top = 0.0f;
  binfo.width = 0.0f;
  binfo.height = 0.0f;
  binfo.detectionConfidence = 0.0f;
  binfo.mask = nullptr;

  const float maxProb = outputTensor[rowIndex * 7 + 4];
  const int maxIndex = static_cast<int>(outputTensor[rowIndex * 7 + 5]);

  if (maxProb < minPreclusterThreshold) {
    return decoded;
  }

  const float bxc = outputTensor[rowIndex * 7 + 0];
  const float byc = outputTensor[rowIndex * 7 + 1];
  const float bw = outputTensor[rowIndex * 7 + 2];
  const float bh = outputTensor[rowIndex * 7 + 3];
  const float angle = outputTensor[rowIndex * 7 + 6];

  decoded.angle = angle;

  const float halfW = bw * 0.5f;
  const float halfH = bh * 0.5f;
  const float cosA = std::cos(angle);
  const float sinA = std::sin(angle);

  const float localX[kObbPointCount] = {-halfW, halfW, halfW, -halfW};
  const float localY[kObbPointCount] = {-halfH, -halfH, halfH, halfH};

  for (unsigned int point = 0; point < kObbPointCount; ++point) {
    float cornerX = bxc + localX[point] * cosA - localY[point] * sinA;
    float cornerY = byc + localX[point] * sinA + localY[point] * cosA;

    cornerX = clampFloat(cornerX, 0.0f, static_cast<float>(netW));
    cornerY = clampFloat(cornerY, 0.0f, static_cast<float>(netH));

    decoded.corners[point * 2 + 0] = cornerX;
    decoded.corners[point * 2 + 1] = cornerY;
  }

  binfo.left = clampFloat(bxc - halfW, 0.0f, static_cast<float>(netW));
  binfo.top = clampFloat(byc - halfH, 0.0f, static_cast<float>(netH));
  binfo.width = std::min(static_cast<float>(netW), std::max(0.0f, bxc + halfW - binfo.left));
  binfo.height = std::min(static_cast<float>(netH), std::max(0.0f, byc + halfH - binfo.top));
  binfo.detectionConfidence = maxProb;
  binfo.classId = maxIndex;

  return decoded;
}

static bool NvDsInferParseCustomYoloOBB(std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo, NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferInstanceMaskInfo>& objectList)
{
  if (outputLayersInfo.empty()) {
    std::cerr << "ERROR: Could not find output layer in bbox parsing" << std::endl;
    return false;
  }

  const NvDsInferLayerInfo& outputTensor = outputLayersInfo[0];
  const unsigned int outputSize = static_cast<unsigned int>(outputTensor.inferDims.d[0]);
  const float* outputBuffer = static_cast<const float*>(outputTensor.buffer);

  const float minPreclusterThreshold = *(std::min_element(detectionParams.perClassPreclusterThreshold.begin(),
      detectionParams.perClassPreclusterThreshold.end()));

  objectList.clear();
  objectList.reserve(outputSize);

  for (unsigned int row = 0; row < outputSize; ++row) {
    const DecodedYoloOBBInfo decodedObject = decodeTensorYoloOBB(
        outputBuffer, row, networkInfo.width, networkInfo.height, minPreclusterThreshold);

    if (!isValidYoloOBBInfo(decodedObject)) {
      continue;
    }

    NvDsInferInstanceMaskInfo object = decodedObject.object;
    object.mask = new float[kObbMaskElementCount];
    for (unsigned int point = 0; point < kObbPointCount; ++point) {
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

static bool NvDsInferParseCustomYolo11OBB(std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo, NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferInstanceMaskInfo>& objectList)
{
  if (outputLayersInfo.empty()) {
    std::cerr << "ERROR: Could not find output layer in bbox parsing" << std::endl;
    return false;
  }

  const NvDsInferLayerInfo& outputTensor = outputLayersInfo[0];
  const unsigned int outputSize = static_cast<unsigned int>(outputTensor.inferDims.d[0]);
  const float* outputBuffer = static_cast<const float*>(outputTensor.buffer);

  const float minPreclusterThreshold = *(std::min_element(detectionParams.perClassPreclusterThreshold.begin(),
      detectionParams.perClassPreclusterThreshold.end()));

  std::vector<DecodedYoloOBBInfo> decodedObjects;
  decodedObjects.reserve(outputSize);

  for (unsigned int row = 0; row < outputSize; ++row) {
    const DecodedYoloOBBInfo decodedObject = decodeTensorYoloOBB(
        outputBuffer, row, networkInfo.width, networkInfo.height, minPreclusterThreshold);

    if (!isValidYoloOBBInfo(decodedObject)) {
      continue;
    }

    decodedObjects.push_back(decodedObject);
  }

  decodedObjects = nonMaximumSuppression(decodedObjects);

  objectList.clear();
  objectList.reserve(decodedObjects.size());

  for (const DecodedYoloOBBInfo& decodedObject : decodedObjects) {
    NvDsInferInstanceMaskInfo object = decodedObject.object;
    object.mask = new float[kObbMaskElementCount];
    for (unsigned int point = 0; point < kObbPointCount; ++point) {
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
NvDsInferParseYoloOBB(std::vector<NvDsInferLayerInfo> const& outputLayersInfo, NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams, std::vector<NvDsInferInstanceMaskInfo>& objectList)
{
  return NvDsInferParseCustomYoloOBB(outputLayersInfo, networkInfo, detectionParams, objectList);
}

extern "C" bool
NvDsInferParseYolo11OBB(std::vector<NvDsInferLayerInfo> const& outputLayersInfo, NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams, std::vector<NvDsInferInstanceMaskInfo>& objectList)
{
  return NvDsInferParseCustomYolo11OBB(outputLayersInfo, networkInfo, detectionParams, objectList);
}

CHECK_CUSTOM_INSTANCE_MASK_PARSE_FUNC_PROTOTYPE(NvDsInferParseYoloOBB);
CHECK_CUSTOM_INSTANCE_MASK_PARSE_FUNC_PROTOTYPE(NvDsInferParseYolo11OBB);
