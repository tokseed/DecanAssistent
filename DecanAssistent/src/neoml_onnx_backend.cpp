#include "neoml_onnx_backend.h"

#include <opencv2/imgproc.hpp>

#if defined(DECAN_WITH_NEOML)
#include <opencv2/dnn.hpp>
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <utility>

#if defined(DECAN_WITH_NEOML)
#include <NeoOnnx/NeoOnnxImport.h>
#include <NeoML/NeoML.h>
#endif

namespace decan {
namespace {

std::string modelLabel(const std::string& path, const char* kind) {
    return std::string("NeoML ") + kind + " model (" + path + ")";
}

#if defined(DECAN_WITH_NEOML)

struct TensorResult {
    std::vector<float> values;
    std::array<int, 7> dimensions{};

    struct InputTransform {
        bool letterboxed = false;
        float scale = 1.0F;
        float scaleX = 1.0F;
        float scaleY = 1.0F;
        int paddingLeft = 0;
        int paddingTop = 0;
        int inputWidth = 0;
        int inputHeight = 0;
    } inputTransform;
};

struct InputSettings {
    std::string inputName;
    std::string outputName;
    int inputWidth = 0;
    int inputHeight = 0;
    int inputChannels = 0;
    bool channelsFirst = true;
    bool inputIsBgr = false;
    float inputScale = 1.0F / 255.0F;
    std::vector<float> mean;
    std::vector<float> standardDeviation;
    cv::Size inputResolution{0, 0};
    bool letterbox = false;
};

// NeoOnnx::CTensorLayout::IOLayout сохраняет порядок осей ONNX, но размещает
// их в семантических измерениях CBlobDesc (например, для 4D: BatchWidth,
// Height, Width, Channels). Внутри runtime всегда работаем с логическими
// ONNX-размерами, а не с индексами полей CBlobDesc.
std::array<int, 7> logicalTensorDimensions(const NeoML::CBlobDesc& description) {
    int physical[NeoML::CBlobDesc::MaxDimensions]{};
    description.GetDimSizes(physical);
    static constexpr std::array<std::array<int, 7>, 7> positions = {{
        {{6, -1, -1, -1, -1, -1, -1}},
        {{1, 6, -1, -1, -1, -1, -1}},
        {{0, 1, 6, -1, -1, -1, -1}},
        {{1, 3, 4, 6, -1, -1, -1}},
        {{0, 1, 3, 4, 6, -1, -1}},
        {{0, 1, 3, 4, 5, 6, -1}},
        {{0, 1, 2, 3, 4, 5, 6}}
    }};

    for (int rank = 1; rank <= NeoML::CBlobDesc::MaxDimensions; ++rank) {
        std::array<int, 7> logical{};
        logical.fill(1);
        std::array<bool, 7> used{};
        bool valid = true;
        for (int axis = 0; axis < rank; ++axis) {
            const int physicalAxis = positions[static_cast<std::size_t>(rank - 1)][static_cast<std::size_t>(axis)];
            logical[static_cast<std::size_t>(axis)] = physical[physicalAxis];
            used[static_cast<std::size_t>(physicalAxis)] = true;
        }
        for (int axis = 0; axis < NeoML::CBlobDesc::MaxDimensions; ++axis) {
            if (!used[static_cast<std::size_t>(axis)] && physical[axis] != 1) {
                valid = false;
                break;
            }
        }
        if (valid) return logical;
    }

    std::array<int, 7> fallback{};
    std::copy(std::begin(physical), std::end(physical), fallback.begin());
    return fallback;
}

InputSettings inputSettings(const NeoMLOcrModelOptions& options) {
    InputSettings settings;
    settings.inputName = options.inputName;
    settings.outputName = options.outputName;
    settings.inputWidth = options.inputWidth;
    settings.inputHeight = options.inputHeight;
    settings.inputChannels = options.inputChannels;
    settings.channelsFirst = options.channelsFirst;
    settings.inputIsBgr = options.inputIsBgr;
    settings.inputScale = options.inputScale;
    settings.mean = options.mean;
    settings.standardDeviation = options.standardDeviation;
    return settings;
}

InputSettings inputSettings(const NeoMLDetectorModelOptions& options) {
    InputSettings settings;
    settings.inputName = options.inputName;
    settings.outputName = options.outputName;
    settings.inputWidth = options.inputWidth;
    settings.inputHeight = options.inputHeight;
    settings.inputChannels = options.inputChannels;
    settings.channelsFirst = options.channelsFirst;
    settings.inputIsBgr = options.inputIsBgr;
    settings.inputScale = options.inputScale;
    settings.mean = options.mean;
    settings.standardDeviation = options.standardDeviation;
    settings.inputResolution = options.inputResolution;
    settings.letterbox = options.letterbox;
    return settings;
}

class OnnxRuntime final {
public:
    OnnxRuntime(const std::string& path, InputSettings settings)
        : settings_(std::move(settings)),
          mathEngine_(NeoML::GetDefaultCpuMathEngine()),
          random_(0xDEC0A11u),
          dnn_(std::make_unique<NeoML::CDnn>(random_, mathEngine_)) {
        if (path.empty()) throw std::invalid_argument("путь к ONNX-модели пуст");
        if (!std::filesystem::is_regular_file(path)) {
            throw std::runtime_error("файл ONNX-модели не найден: " + path);
        }

        NeoOnnx::CImportSettings importSettings;
        NeoOnnx::CImportedModelInfo info;
        NeoOnnx::LoadFromOnnx(path.c_str(), importSettings, *dnn_, info);
        if (info.Inputs.Size() == 0 || info.Outputs.Size() == 0) {
            throw std::runtime_error("ONNX-граф не содержит входа или выхода: " + path);
        }

        inputName_ = settings_.inputName.empty()
            ? std::string(info.Inputs[0].Name)
            : settings_.inputName;
        outputName_ = settings_.outputName.empty()
            ? std::string(info.Outputs[0].Name)
            : settings_.outputName;

        const FObj::CPtr<NeoML::CBaseLayer> inputLayer = dnn_->GetLayer(inputName_.c_str());
        const FObj::CPtr<NeoML::CBaseLayer> outputLayer = dnn_->GetLayer(outputName_.c_str());
        if (inputLayer == nullptr) throw std::runtime_error("входной слой не найден: " + inputName_);
        if (outputLayer == nullptr) throw std::runtime_error("выходной слой не найден: " + outputName_);

        source_ = dynamic_cast<NeoML::CSourceLayer*>(inputLayer.Ptr());
        sink_ = dynamic_cast<NeoML::CSinkLayer*>(outputLayer.Ptr());
        if (source_ == nullptr || sink_ == nullptr) {
            throw std::runtime_error("ONNX input/output не сопоставлены с Source/Sink слоями NeoML");
        }

        if (source_->GetBlob() != nullptr) {
            inputDescription_ = source_->GetBlob()->GetDesc();
        } else {
            const int channels = std::max(1, settings_.inputChannels);
            const int height = std::max(1, settings_.inputHeight > 0
                ? settings_.inputHeight : settings_.inputResolution.height);
            const int width = std::max(1, settings_.inputWidth > 0
                ? settings_.inputWidth : settings_.inputResolution.width);
            inputDescription_ = NeoML::CBlobDesc(NeoML::CT_Float);
            inputDescription_.SetDimSize(NeoML::BD_BatchWidth, 1);
            inputDescription_.SetDimSize(NeoML::BD_Height, settings_.channelsFirst ? channels : height);
            inputDescription_.SetDimSize(NeoML::BD_Width, settings_.channelsFirst ? height : width);
            inputDescription_.SetDimSize(NeoML::BD_Channels, settings_.channelsFirst ? width : channels);
        }
        updateInputGeometry();
    }

    TensorResult infer(const cv::Mat& image) {
        if (image.empty()) throw std::invalid_argument("NeoML получил пустое изображение");

        const auto input = FObj::CPtr<NeoML::CDnnBlob>(
            NeoML::CDnnBlob::CreateBlob(mathEngine_, NeoML::CT_Float, inputDescription_));
        std::vector<float> values(static_cast<std::size_t>(input->GetDataSize()), 0.0F);
        TensorResult::InputTransform inputTransform;
        fillInput(image, values, inputTransform);
        input->CopyFrom(values.data());
        source_->SetBlob(input);
        dnn_->RunOnce();

        const FObj::CPtr<NeoML::CDnnBlob>& output = sink_->GetBlob();
        if (output == nullptr) throw std::runtime_error("NeoML output blob не создан");
        TensorResult result;
        result.dimensions = logicalTensorDimensions(output->GetDesc());
        result.values.resize(static_cast<std::size_t>(output->GetDataSize()));
        if (output->GetDataType() == NeoML::CT_Float) {
            output->CopyTo(result.values.data());
        } else if (output->GetDataType() == NeoML::CT_Int) {
            std::vector<int> integers(result.values.size());
            output->CopyTo(integers.data());
            std::transform(integers.begin(), integers.end(), result.values.begin(),
                           [](int value) { return static_cast<float>(value); });
        } else {
            throw std::runtime_error("NeoML output имеет неподдерживаемый тип данных");
        }
        result.inputTransform = inputTransform;
        return result;
    }

    int inputWidth() const { return inputWidth_; }
    int inputHeight() const { return inputHeight_; }

private:
    void updateInputGeometry() {
        const std::array<int, 7> dimensions = logicalTensorDimensions(inputDescription_);
        int rank = static_cast<int>(dimensions.size());
        while (rank > 1 && dimensions[static_cast<std::size_t>(rank - 1)] == 1) --rank;

        if (rank >= 4) {
            if (settings_.channelsFirst) {
                inputChannels_ = dimensions[1];
                inputHeight_ = dimensions[2];
                inputWidth_ = dimensions[3];
            } else {
                inputHeight_ = dimensions[1];
                inputWidth_ = dimensions[2];
                inputChannels_ = dimensions[3];
            }
        } else if (rank == 3) {
            if (settings_.channelsFirst) {
                inputChannels_ = dimensions[0];
                inputHeight_ = dimensions[1];
                inputWidth_ = dimensions[2];
            } else {
                inputHeight_ = dimensions[0];
                inputWidth_ = dimensions[1];
                inputChannels_ = dimensions[2];
            }
        } else {
            inputChannels_ = 1;
            inputHeight_ = dimensions[0];
            inputWidth_ = dimensions[1];
        }

        if (settings_.inputChannels > 0 && inputChannels_ <= 1) inputChannels_ = settings_.inputChannels;
        if (inputHeight_ <= 1) {
            inputHeight_ = settings_.inputHeight > 0
                ? settings_.inputHeight : settings_.inputResolution.height;
        }
        if (inputWidth_ <= 1) {
            inputWidth_ = settings_.inputWidth > 0
                ? settings_.inputWidth : settings_.inputResolution.width;
        }
        if (inputWidth_ <= 0 || inputHeight_ <= 0 || inputChannels_ <= 0) {
            throw std::runtime_error("не удалось определить размер входа ONNX-модели");
        }
    }

    float channelValue(const std::vector<float>& values, int channel, float defaultValue) const {
        if (values.empty()) return defaultValue;
        if (channel < static_cast<int>(values.size())) return values[static_cast<std::size_t>(channel)];
        return values.back();
    }

    void fillInput(const cv::Mat& image, std::vector<float>& destination,
                   TensorResult::InputTransform& transform) const {
        cv::Mat converted;
        if (inputChannels_ == 1) {
            if (image.channels() == 1) converted = image;
            else if (image.channels() == 4) cv::cvtColor(image, converted, cv::COLOR_BGRA2GRAY);
            else cv::cvtColor(image, converted, cv::COLOR_BGR2GRAY);
        } else {
            if (image.channels() == 4) cv::cvtColor(image, converted, cv::COLOR_BGRA2BGR);
            else if (image.channels() == 1) cv::cvtColor(image, converted, cv::COLOR_GRAY2BGR);
            else converted = image;
            if (!settings_.inputIsBgr) cv::cvtColor(converted, converted, cv::COLOR_BGR2RGB);
        }

        if (converted.empty() || converted.cols <= 0 || converted.rows <= 0) {
            throw std::runtime_error("не удалось подготовить изображение для NeoML input");
        }

        cv::Mat resized;
        transform.inputWidth = inputWidth_;
        transform.inputHeight = inputHeight_;
        if (settings_.letterbox) {
            const float scale = std::min(
                static_cast<float>(inputWidth_) / static_cast<float>(converted.cols),
                static_cast<float>(inputHeight_) / static_cast<float>(converted.rows));
            if (!std::isfinite(scale) || scale <= 0.0F) {
                throw std::runtime_error("некорректный масштаб letterbox");
            }
            const int resizedWidth = std::max(1, static_cast<int>(std::lround(
                static_cast<float>(converted.cols) * scale)));
            const int resizedHeight = std::max(1, static_cast<int>(std::lround(
                static_cast<float>(converted.rows) * scale)));
            cv::resize(converted, resized, cv::Size(resizedWidth, resizedHeight), 0.0, 0.0,
                       cv::INTER_LINEAR);

            const int paddingWidth = inputWidth_ - resized.cols;
            const int paddingHeight = inputHeight_ - resized.rows;
            if (paddingWidth < 0 || paddingHeight < 0) {
                throw std::runtime_error("letterbox вышел за размер input blob");
            }
            transform.letterboxed = true;
            transform.scaleX = static_cast<float>(resized.cols) /
                               static_cast<float>(converted.cols);
            transform.scaleY = static_cast<float>(resized.rows) /
                               static_cast<float>(converted.rows);
            // scale оставляем для совместимости с общей проверкой валидности;
            // для обратного преобразования используются точные scaleX/scaleY.
            transform.scale = std::min(transform.scaleX, transform.scaleY);
            transform.paddingLeft = paddingWidth / 2;
            transform.paddingTop = paddingHeight / 2;
            cv::Mat canvas(inputHeight_, inputWidth_, resized.type(), cv::Scalar::all(114));
            resized.copyTo(canvas(cv::Rect(transform.paddingLeft, transform.paddingTop,
                                           resized.cols, resized.rows)));
            resized = std::move(canvas);
        } else {
            cv::resize(converted, resized, cv::Size(inputWidth_, inputHeight_), 0.0, 0.0,
                       cv::INTER_AREA);
            transform.letterboxed = false;
            transform.scale = 1.0F;
            transform.scaleX = static_cast<float>(inputWidth_) /
                               static_cast<float>(converted.cols);
            transform.scaleY = static_cast<float>(inputHeight_) /
                               static_cast<float>(converted.rows);
            transform.paddingLeft = 0;
            transform.paddingTop = 0;
        }
        if (resized.depth() != CV_8U && resized.depth() != CV_32F) {
            resized.convertTo(resized, CV_32F);
        }

        const int channels = std::min(inputChannels_, resized.channels());
        const std::size_t expected = static_cast<std::size_t>(inputWidth_) *
            static_cast<std::size_t>(inputHeight_) * static_cast<std::size_t>(inputChannels_);
        if (destination.size() < expected) {
            throw std::runtime_error("размер NeoML input blob меньше размера изображения");
        }

        auto pixel = [&](int y, int x, int channel) {
            if (resized.depth() == CV_32F) {
                if (resized.channels() == 1) return resized.at<float>(y, x);
                return resized.ptr<float>(y)[x * resized.channels() + channel];
            }
            if (resized.channels() == 1) return static_cast<float>(resized.at<unsigned char>(y, x));
            return static_cast<float>(resized.ptr<unsigned char>(y)[x * resized.channels() + channel]);
        };

        std::fill(destination.begin(), destination.end(), 0.0F);
        for (int y = 0; y < inputHeight_; ++y) {
            for (int x = 0; x < inputWidth_; ++x) {
                for (int channel = 0; channel < inputChannels_; ++channel) {
                    const int sourceChannel = std::min(channel, std::max(0, channels - 1));
                    const float raw = pixel(y, x, sourceChannel) * settings_.inputScale;
                    const float mean = channelValue(settings_.mean, channel, 0.0F);
                    const float deviation = channelValue(settings_.standardDeviation, channel, 1.0F);
                    const float normalized = (deviation == 0.0F ? raw - mean : (raw - mean) / deviation);
                    const std::size_t index = settings_.channelsFirst
                        ? (static_cast<std::size_t>(channel) * inputHeight_ + y) * inputWidth_ + x
                        : (static_cast<std::size_t>(y) * inputWidth_ + x) * inputChannels_ + channel;
                    if (index < destination.size()) destination[index] = normalized;
                }
            }
        }
    }

    InputSettings settings_;
    NeoML::IMathEngine& mathEngine_;
    NeoML::CRandom random_;
    std::unique_ptr<NeoML::CDnn> dnn_;
    FObj::CPtr<NeoML::CSourceLayer> source_;
    FObj::CPtr<NeoML::CSinkLayer> sink_;
    NeoML::CBlobDesc inputDescription_;
    std::string inputName_;
    std::string outputName_;
    int inputWidth_ = 0;
    int inputHeight_ = 0;
    int inputChannels_ = 0;
};

int logicalRank(const std::array<int, 7>& dimensions) {
    int rank = static_cast<int>(dimensions.size());
    while (rank > 1 && dimensions[static_cast<std::size_t>(rank - 1)] == 1) --rank;
    return rank;
}

std::string decodeCtc(const TensorResult& output, const NeoMLOcrModelOptions& options,
                      float& confidence) {
    const int rank = logicalRank(output.dimensions);
    if (rank < 1 || rank > 3) throw std::runtime_error("неподдерживаемая размерность OCR output");
    const int classes = options.outputClasses > 0
        ? options.outputClasses
        : (rank == 1 || rank == 2 || options.classesLast
            ? output.dimensions[static_cast<std::size_t>(rank - 1)]
            : output.dimensions[static_cast<std::size_t>(rank - 2)]);
    const int time = rank == 1 ? 1 : rank == 2
        ? (options.classesLast ? output.dimensions[0] : output.dimensions[1])
        : (options.classesLast
            ? output.dimensions[static_cast<std::size_t>(rank - 2)]
            : output.dimensions[static_cast<std::size_t>(rank - 1)]);
    if (classes <= 1 || time <= 0 || static_cast<std::size_t>(classes) * time > output.values.size()) {
        throw std::runtime_error("размер OCR output не совпадает с количеством классов");
    }
    if (options.alphabet.empty()) throw std::runtime_error("для OCR-модели не задан alphabet");
    if (options.alphabetIncludesBlank) {
        if (static_cast<int>(options.alphabet.size()) != classes) {
            throw std::runtime_error("alphabet OCR не совпадает с количеством классов");
        }
    } else if (static_cast<int>(options.alphabet.size()) != classes - 1) {
        throw std::runtime_error("alphabet OCR должен содержать classes - 1 символов");
    }
    if (options.blankIndex < 0 || options.blankIndex >= classes) {
        throw std::runtime_error("blankIndex OCR находится вне диапазона");
    }

    std::string text;
    int previous = options.blankIndex;
    double confidenceSum = 0.0;
    int confidenceCount = 0;
    for (int t = 0; t < time; ++t) {
        const auto valueAt = [&](int c) {
            if (rank == 1 || options.classesLast) {
                return output.values[static_cast<std::size_t>(t) * classes + c];
            }
            return output.values[static_cast<std::size_t>(c) * time + t];
        };

        float maxLogit = -std::numeric_limits<float>::infinity();
        int bestClass = 0;
        for (int c = 0; c < classes; ++c) {
            const float logit = valueAt(c);
            if (logit > maxLogit) {
                maxLogit = logit;
                bestClass = c;
            }
        }
        float denominator = 0.0F;
        for (int c = 0; c < classes; ++c) denominator += std::exp(valueAt(c) - maxLogit);
        const float probability = denominator > 0.0F ? 1.0F / denominator : 0.0F;
        confidenceSum += probability;
        ++confidenceCount;

        if (bestClass == options.blankIndex || bestClass == previous) {
            previous = bestClass;
            continue;
        }
        int alphabetIndex = bestClass;
        if (!options.alphabetIncludesBlank && bestClass > options.blankIndex) --alphabetIndex;
        if (alphabetIndex >= 0 && alphabetIndex < static_cast<int>(options.alphabet.size())) {
            text += options.alphabet[static_cast<std::size_t>(alphabetIndex)];
        }
        previous = bestClass;
    }
    confidence = confidenceCount == 0 ? 0.0F
        : static_cast<float>(confidenceSum / confidenceCount);
    return text;
}

std::vector<int> detectorDimensions(const TensorResult& output, bool hasBatchIndex) {
    const int rank = logicalRank(output.dimensions);
    if (rank < 1 || rank > 3) throw std::runtime_error("неподдерживаемая размерность detector output");
    if (rank == 1) {
        const int columns = static_cast<int>(output.values.size());
        if (columns == 5 || columns == 6 || columns == 7) return {1, columns};
        throw std::runtime_error("одномерный detector output должен содержать 5, 6 или 7 значений");
    }
    const int columns = output.dimensions[static_cast<std::size_t>(rank - 1)];
    const int rows = static_cast<int>(output.values.size()) / std::max(1, columns);
    const int expectedColumns = hasBatchIndex ? 7 : 6;
    if (columns != 5 && columns != 6 && columns != 7 && columns != expectedColumns) {
        throw std::runtime_error("detector output должен иметь 5, 6 или 7 колонок");
    }
    if (rows <= 0 || static_cast<std::size_t>(rows * columns) > output.values.size()) {
        throw std::runtime_error("пустой detector output");
    }
    return {rows, columns};
}

#endif // DECAN_WITH_NEOML

} // namespace

#if defined(DECAN_WITH_NEOML)

struct NeoMLOnnxBackend::Impl {
    struct OcrModel {
        OnnxRuntime runtime;
        NeoMLOcrModelOptions options;

        OcrModel(const NeoMLOcrModelOptions& modelOptions)
            : runtime(modelOptions.path, inputSettings(modelOptions)), options(modelOptions) {}
    };

    struct DetectorModel {
        OnnxRuntime runtime;
        NeoMLDetectorModelOptions options;

        DetectorModel(const NeoMLDetectorModelOptions& modelOptions)
            : runtime(modelOptions.path, inputSettings(modelOptions)), options(modelOptions) {}
    };

    explicit Impl(const NeoMLModelConfiguration& configuration) {
        loadModel(configuration.printed, "printed OCR", printed);
        loadModel(configuration.handwriting, "handwriting", handwriting);
        loadModel(configuration.sealStamp, "seal/stamp detector", sealStamp);
        loadModel(configuration.coatOfArms, "coat-of-arms detector", coatOfArms);
    }

    template<class Options, class Model>
    void loadModel(const Options& options, const char* kind, std::unique_ptr<Model>& destination) {
        if (options.path.empty()) {
            diagnostics.push_back(modelLabel("<empty path>", kind) + " не настроена; используется fallback");
            return;
        }
        try {
            destination = std::make_unique<Model>(options);
        } catch (const std::exception& error) {
            diagnostics.push_back(modelLabel(options.path, kind) + " не загружена: " + error.what());
        }
    }

    NeoMLTextResult recognizeText(const cv::Mat& image) {
        if (!printed) throw std::runtime_error("NeoML printed OCR model is not configured");
        float confidence = 0.0F;
        const TensorResult output = printed->runtime.infer(image);
        return {decodeCtc(output, printed->options, confidence), confidence};
    }

    NeoMLTextResult recognizeHandwritten(const cv::Mat& image, HandwrittenField field) {
        (void)field;
        if (!handwriting) throw std::runtime_error("NeoML handwriting OCR model is not configured");
        float confidence = 0.0F;
        const TensorResult output = handwriting->runtime.infer(image);
        return {decodeCtc(output, handwriting->options, confidence), confidence};
    }

    std::vector<VisualMark> detectVisualMarks(const cv::Mat& image) {
        if (image.empty() || image.cols <= 0 || image.rows <= 0) {
            throw std::invalid_argument("NeoML detector получил пустое изображение");
        }
        std::vector<VisualMark> result;
        bool successful = false;
        if (sealStamp) successful = runDetector(*sealStamp, image, result) || successful;
        if (coatOfArms) successful = runDetector(*coatOfArms, image, result) || successful;
        if (!successful) throw std::runtime_error("NeoML detector models are not configured");

        std::sort(result.begin(), result.end(), [](const VisualMark& left, const VisualMark& right) {
            return left.confidence > right.confidence;
        });
        return result;
    }

    template<class Model>
    bool runDetector(Model& model, const cv::Mat& image, std::vector<VisualMark>& result) {
        try {
            const TensorResult output = model.runtime.infer(image);
            const auto shape = detectorDimensions(output, model.options.outputHasBatchIndex);
            const int rows = shape[0];
            const int columns = shape[1];
            std::vector<VisualMark> candidates;
            candidates.reserve(static_cast<std::size_t>(rows));
            for (int row = 0; row < rows; ++row) {
                const std::size_t offset = static_cast<std::size_t>(row * columns);
                const int coordinateOffset = columns == 7 ? 1 : 0;
                const int scoreIndex = coordinateOffset + 4;
                const int classIndex = coordinateOffset + 5;
                const float confidence = output.values[offset + scoreIndex];
                if (!std::isfinite(confidence) || confidence < model.options.confidenceThreshold) continue;

                float x = output.values[offset + coordinateOffset];
                float y = output.values[offset + coordinateOffset + 1];
                float width = output.values[offset + coordinateOffset + 2];
                float height = output.values[offset + coordinateOffset + 3];
                if (!model.options.xywh) {
                    width -= x;
                    height -= y;
                } else if (model.options.centerCoordinates) {
                    x -= width * 0.5F;
                    y -= height * 0.5F;
                }
                const float x2 = x + width;
                const float y2 = y + height;
                if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(x2) ||
                    !std::isfinite(y2) || x2 <= x || y2 <= y) {
                    continue;
                }

                // Сначала переводим координаты в пиксели tensor input, затем
                // отменяем letterbox и только после этого возвращаемся к
                // размерам исходной страницы. Прямое умножение normalized
                // координат на image.cols/rows неверно при наличии padding.
                const float inputWidth = static_cast<float>(
                    std::max(1, model.runtime.inputWidth()));
                const float inputHeight = static_cast<float>(
                    std::max(1, model.runtime.inputHeight()));
                float modelLeft = x;
                float modelTop = y;
                float modelRight = x2;
                float modelBottom = y2;
                if (model.options.coordinatesNormalized) {
                    modelLeft *= inputWidth;
                    modelRight *= inputWidth;
                    modelTop *= inputHeight;
                    modelBottom *= inputHeight;
                }

                float originalLeft = 0.0F;
                float originalTop = 0.0F;
                float originalRight = 0.0F;
                float originalBottom = 0.0F;
                const auto& transform = output.inputTransform;
                if (transform.letterboxed) {
                    if (!std::isfinite(transform.scaleX) || !std::isfinite(transform.scaleY) ||
                        transform.scaleX <= 0.0F || transform.scaleY <= 0.0F) {
                        throw std::runtime_error("NeoML detector получил некорректный letterbox scale");
                    }
                    originalLeft = (modelLeft - static_cast<float>(transform.paddingLeft)) /
                                   transform.scaleX;
                    originalTop = (modelTop - static_cast<float>(transform.paddingTop)) /
                                  transform.scaleY;
                    originalRight = (modelRight - static_cast<float>(transform.paddingLeft)) /
                                    transform.scaleX;
                    originalBottom = (modelBottom - static_cast<float>(transform.paddingTop)) /
                                     transform.scaleY;
                } else {
                    originalLeft = modelLeft * static_cast<float>(image.cols) / inputWidth;
                    originalTop = modelTop * static_cast<float>(image.rows) / inputHeight;
                    originalRight = modelRight * static_cast<float>(image.cols) / inputWidth;
                    originalBottom = modelBottom * static_cast<float>(image.rows) / inputHeight;
                }

                const int left = std::clamp(static_cast<int>(std::lround(originalLeft)), 0, image.cols);
                const int top = std::clamp(static_cast<int>(std::lround(originalTop)), 0, image.rows);
                const int right = std::clamp(static_cast<int>(std::lround(originalRight)), left, image.cols);
                const int bottom = std::clamp(static_cast<int>(std::lround(originalBottom)), top, image.rows);
                if (right <= left || bottom <= top) continue;

                VisualMark::Type type = model.options.forcedType;
                if (type == VisualMark::Type::Unknown && !model.options.classTypes.empty() && columns >= 6) {
                    const int classId = static_cast<int>(std::lround(output.values[offset + classIndex]));
                    if (classId >= 0 && classId < static_cast<int>(model.options.classTypes.size())) {
                        type = model.options.classTypes[static_cast<std::size_t>(classId)];
                    }
                }
                if (type == VisualMark::Type::Unknown) type = VisualMark::Type::Seal;
                candidates.push_back({type, {left, top, right - left, bottom - top},
                                     std::clamp(confidence, 0.0F, 1.0F)});
            }

            // NMS выполняется отдельно для каждого класса: герб и печать,
            // даже если они пересекаются, не должны подавлять друг друга.
            std::vector<VisualMark::Type> types;
            for (const VisualMark& candidate : candidates) {
                if (std::find(types.begin(), types.end(), candidate.type) == types.end()) {
                    types.push_back(candidate.type);
                }
            }
            const float scoreThreshold = std::clamp(model.options.confidenceThreshold,
                                                     0.0F, 1.0F);
            const float nmsThreshold = std::clamp(model.options.nmsIouThreshold,
                                                  0.0F, 1.0F);
            for (const VisualMark::Type type : types) {
                std::vector<cv::Rect> boxes;
                std::vector<float> scores;
                std::vector<std::size_t> candidateIndices;
                for (std::size_t index = 0; index < candidates.size(); ++index) {
                    if (candidates[index].type != type) continue;
                    const auto& bounds = candidates[index].bounds;
                    boxes.emplace_back(bounds.x, bounds.y, bounds.width, bounds.height);
                    scores.push_back(candidates[index].confidence);
                    candidateIndices.push_back(index);
                }
                std::vector<int> kept;
                cv::dnn::NMSBoxes(boxes, scores, scoreThreshold, nmsThreshold, kept);
                for (const int localIndex : kept) {
                    if (localIndex >= 0 && localIndex < static_cast<int>(candidateIndices.size())) {
                        result.push_back(candidates[candidateIndices[static_cast<std::size_t>(localIndex)]]);
                    }
                }
            }
            return true;
        } catch (const std::exception& error) {
            diagnostics.push_back(modelLabel(model.options.path, "detector") +
                                  " не отработала: " + error.what());
            return false;
        }
    }

    std::unique_ptr<OcrModel> printed;
    std::unique_ptr<OcrModel> handwriting;
    std::unique_ptr<DetectorModel> sealStamp;
    std::unique_ptr<DetectorModel> coatOfArms;
    std::vector<std::string> diagnostics;
};

#else

struct NeoMLOnnxBackend::Impl {
    explicit Impl(const NeoMLModelConfiguration& configuration) {
        add(configuration.printed.path, "printed OCR");
        add(configuration.handwriting.path, "handwriting");
        add(configuration.sealStamp.path, "seal/stamp detector");
        add(configuration.coatOfArms.path, "coat-of-arms detector");
        diagnostics.push_back("NeoML ONNX backend не включён в эту сборку; используется fallback");
    }

    void add(const std::string& path, const char* kind) {
        if (path.empty()) {
            diagnostics.push_back(modelLabel("<empty path>", kind) + " не настроена");
        } else if (!std::filesystem::is_regular_file(path)) {
            diagnostics.push_back(modelLabel(path, kind) + " не найдена");
        }
    }

    std::vector<std::string> diagnostics;
};

#endif

NeoMLOnnxBackend::NeoMLOnnxBackend(const NeoMLModelConfiguration& configuration)
    : impl_(std::make_unique<Impl>(configuration)) {}

NeoMLOnnxBackend::~NeoMLOnnxBackend() = default;

NeoMLTextResult NeoMLOnnxBackend::recognizeText(const cv::Mat& image, std::size_t page) {
    (void)page;
#if defined(DECAN_WITH_NEOML)
    return impl_->recognizeText(image);
#else
    (void)image;
    throw std::runtime_error("NeoML printed OCR backend is not linked");
#endif
}

NeoMLTextResult NeoMLOnnxBackend::recognizeHandwritten(const cv::Mat& image,
                                                       std::size_t page,
                                                       HandwrittenField field) {
    (void)page;
#if defined(DECAN_WITH_NEOML)
    return impl_->recognizeHandwritten(image, field);
#else
    (void)image;
    (void)field;
    throw std::runtime_error("NeoML handwriting OCR backend is not linked");
#endif
}

std::vector<VisualMark> NeoMLOnnxBackend::detectVisualMarks(const cv::Mat& image,
                                                            std::size_t page) {
    (void)page;
#if defined(DECAN_WITH_NEOML)
    return impl_->detectVisualMarks(image);
#else
    (void)image;
    throw std::runtime_error("NeoML detector backend is not linked");
#endif
}

std::vector<std::string> NeoMLOnnxBackend::diagnostics() const {
    return impl_->diagnostics;
}

} // namespace decan
