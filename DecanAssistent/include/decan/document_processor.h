#pragma once

#include <opencv2/core.hpp>

#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "text_post_processor.h"

namespace decan {

enum class DocumentFormat { Png, Jpeg, Pdf, Docx, Txt };

struct BoundingBox {
    int x = 0;
    int y = 0;
    int width = 0;
    int height = 0;
};

struct VisualMark {
    enum class Type { CoatOfArms, Seal, Stamp, Signature, Unknown };
    Type type = Type::Unknown;
    BoundingBox bounds;
    float confidence = 0.0F;
};

struct OcrRegionResult {
    std::string name;
    BoundingBox bounds;
    std::string text;
    std::string rawText;
    float confidence = 0.0F;
};

struct DocumentAnalysis {
    std::string sourceName;
    DocumentFormat format;
    std::string text;
    std::vector<VisualMark> visualMarks;
    std::vector<OcrRegionResult> regions;
    std::size_t pageCount = 0;
    std::vector<std::string> warnings;

    // JSON намеренно формируется внутри модуля, чтобы потребителю не требовалась
    // дополнительная JSON-библиотека только для передачи результата.
    std::string toJson() const;
};

struct NeoMLTextResult {
    std::string text;
    float confidence = 0.0F;
};

// Параметры входа/декодирования OCR-модели. NeoML импортирует ONNX-граф, но
// не знает, как преобразовать его logits в русский текст, поэтому алфавит
// CTC-модели задаётся явно в конфигурации приложения.
struct NeoMLOcrModelOptions {
    std::string path;
    std::string inputName;
    std::string outputName;
    int inputWidth = 0;   // 0 — взять размер из ONNX-входа.
    int inputHeight = 0;  // 0 — взять размер из ONNX-входа.
    int inputChannels = 0;
    bool channelsFirst = true;
    bool inputIsBgr = false;
    float inputScale = 1.0F / 255.0F;
    std::vector<float> mean;
    std::vector<float> standardDeviation;
    std::vector<std::string> alphabet;
    bool alphabetIncludesBlank = false;
    int blankIndex = 0;
    int outputClasses = 0;
    bool classesLast = true;
};

// Описание detector-модели. Поддерживается распространённый ONNX-выход
// [N, 6]: x1, y1, x2, y2, confidence, class_id и вариант [N, 7] с batch_id.
struct NeoMLDetectorModelOptions {
    std::string path;
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
    bool coordinatesNormalized = true;
    bool xywh = false; // x, y, width, height вместо x1, y1, x2, y2.
    bool centerCoordinates = false; // для xywh: x/y — центр, а не левый верхний угол.
    bool outputHasBatchIndex = false;
    // Ожидаемый размер входа detector-модели. Если ONNX-граф содержит
    // фиксированный размер, фактический размер берётся из его input blob;
    // для динамического входа используется это значение.
    cv::Size inputResolution{640, 640};
    // Letterbox сохраняет соотношение сторон и позволяет корректно вернуть
    // координаты детекции в систему координат исходной страницы.
    bool letterbox = true;
    float confidenceThreshold = 0.45F;
    float nmsIouThreshold = 0.40F;
    VisualMark::Type forcedType = VisualMark::Type::Unknown;
    std::vector<VisualMark::Type> classTypes;
};

struct NeoMLModelConfiguration {
    NeoMLOcrModelOptions handwriting;
    NeoMLDetectorModelOptions sealStamp;
    NeoMLDetectorModelOptions coatOfArms;
    // Добавлен последним для совместимости со старыми aggregate-init вызовами.
    // Печатный OCR вызывается для body/header, рукописный — отдельно для
    // date/document_number, чтобы одна модель не подменяла другую.
    NeoMLOcrModelOptions printed;
};

class NeoMLOnnxBackend;

// Тонкий контракт над конкретной моделью NeoML. В разных сборках NeoML меняются
// классы загрузки сетей, поэтому интеграция с ними изолирована от остального кода.
class INeoMLAnalyzer {
public:
    virtual ~INeoMLAnalyzer() = default;
    virtual NeoMLTextResult recognizeText(const cv::Mat& preparedImage,
                                          std::size_t page) = 0;
    // Рукописные поля используют отдельную модель/декодер. Реализация по
    // умолчанию сохраняет обратную совместимость со старым OCR-адаптером.
    virtual NeoMLTextResult recognizeHandwrittenText(const cv::Mat& preparedImage,
                                                     std::size_t page,
                                                     HandwrittenField field) {
        (void)field;
        return recognizeText(preparedImage, page);
    }
    virtual std::vector<VisualMark> detectVisualMarks(const cv::Mat& image,
                                                      std::size_t page) = 0;
    // Возвращает диагностические сообщения загрузки/запуска моделей. Это
    // позволяет сохранить причину fallback в структурированном ответе.
    virtual std::vector<std::string> diagnostics() const { return {}; }
};

// Удобный адаптер для C++ API NeoML: передайте сюда функции, вызывающие ваши
// загруженные NeoML-сети (OCR и detector). В production-коде эти функции обычно
// захватывают экземпляры NeoML::CDnn и соответствующие входные/выходные слои.
class NeoMLAnalyzerAdapter final : public INeoMLAnalyzer {
public:
    using OcrFunction = std::function<NeoMLTextResult(const cv::Mat&, std::size_t)>;
    using HandwrittenOcrFunction = std::function<NeoMLTextResult(
        const cv::Mat&, std::size_t, HandwrittenField)>;
    using DetectorFunction = std::function<std::vector<VisualMark>(const cv::Mat&, std::size_t)>;

    NeoMLAnalyzerAdapter(OcrFunction ocr, DetectorFunction detector);
    NeoMLAnalyzerAdapter(OcrFunction printedOcr,
                         HandwrittenOcrFunction handwrittenOcr,
                         DetectorFunction detector);
    // Вариант адаптера с реальными ONNX-моделями NeoML. Fallback-функции
    // остаются обязательными: повреждённая или несовместимая модель не должна
    // останавливать обработку всего документа.
    NeoMLAnalyzerAdapter(NeoMLModelConfiguration models,
                         OcrFunction printedFallback,
                         HandwrittenOcrFunction handwrittenFallback,
                         DetectorFunction detectorFallback);
    ~NeoMLAnalyzerAdapter() override;
    NeoMLTextResult recognizeText(const cv::Mat&, std::size_t) override;
    NeoMLTextResult recognizeHandwrittenText(const cv::Mat&, std::size_t,
                                             HandwrittenField) override;
    std::vector<VisualMark> detectVisualMarks(const cv::Mat&, std::size_t) override;
    std::vector<std::string> diagnostics() const override;

private:
    void addDiagnostic(const std::string& message) const;

    OcrFunction ocr_;
    HandwrittenOcrFunction handwrittenOcr_;
    DetectorFunction detector_;
    std::unique_ptr<NeoMLOnnxBackend> onnxBackend_;
    mutable std::vector<std::string> diagnostics_;
};

class DocumentProcessor final {
public:
    struct Options {
        std::size_t maxInputBytes = 100 * 1024 * 1024;
        int adaptiveBlockSize = 31;
        double adaptiveConstant = 11.0;
        double headerFraction = 0.25;
        int bodyBorderCropPixels = 4;
        int bodyMorphologyKernelSize = 2;
        bool removeBodyScanLines = true;
        double printedUpscale = 2.0;
        double headerUpscale = 3.0;
        double handwritingUpscale = 3.0;
        // Нормированные ROI типового бланка: дата слева, номер справа.
        double handwrittenDateX = 0.05;
        double handwrittenDateY = 0.19;
        double handwrittenDateWidth = 0.36;
        double handwrittenDateHeight = 0.10;
        double handwrittenNumberX = 0.73;
        double handwrittenNumberY = 0.19;
        double handwrittenNumberWidth = 0.23;
        double handwrittenNumberHeight = 0.10;
        bool runHeaderPass = true;
        bool runHandwrittenPass = true;
        // PDF — контейнер, OpenCV его не растеризует. Для сканированных PDF
        // передайте callback на Poppler/PDFium, возвращающий изображения страниц.
        std::function<std::vector<cv::Mat>(const std::vector<unsigned char>&)> pdfRasterizer;
    };

    explicit DocumentProcessor(std::shared_ptr<INeoMLAnalyzer> analyzer);
    DocumentProcessor(std::shared_ptr<INeoMLAnalyzer> analyzer, Options options);

    DocumentAnalysis processPath(const std::string& path) const;
    DocumentAnalysis processBuffer(const std::string& fileName,
                                   const std::vector<unsigned char>& bytes) const;

private:
    std::shared_ptr<INeoMLAnalyzer> analyzer_;
    Options options_;
};

const char* toString(DocumentFormat format);
const char* toString(VisualMark::Type type);
const char* toString(HandwrittenField field);

} // namespace decan
