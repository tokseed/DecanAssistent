#include "../include/decan/document_processor.h"
#include "neoml_onnx_backend.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <filesystem>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace decan {
namespace {

std::string lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

DocumentFormat formatFromName(const std::string& name) {
    const auto dot = name.find_last_of('.');
    if (dot == std::string::npos) throw std::invalid_argument("Файл не имеет расширения: " + name);
    const std::string ext = lower(name.substr(dot + 1));
    if (ext == "png") return DocumentFormat::Png;
    if (ext == "jpg" || ext == "jpeg") return DocumentFormat::Jpeg;
    if (ext == "pdf") return DocumentFormat::Pdf;
    if (ext == "docx") return DocumentFormat::Docx;
    if (ext == "txt") return DocumentFormat::Txt;
    throw std::invalid_argument("Неподдерживаемый формат файла: ." + ext);
}

std::string jsonEscape(const std::string& value) {
    std::ostringstream out;
    for (unsigned char c : value) {
        switch (c) {
        case '"': out << "\\\""; break;
        case '\\': out << "\\\\"; break;
        case '\n': out << "\\n"; break;
        case '\r': out << "\\r"; break;
        case '\t': out << "\\t"; break;
        default: if (c < 0x20) out << ' '; else out << c;
        }
    }
    return out.str();
}

std::string readText(const std::vector<unsigned char>& bytes) {
    return std::string(reinterpret_cast<const char*>(bytes.data()), bytes.size());
}

void validateSignature(DocumentFormat format, const std::vector<unsigned char>& bytes) {
    const auto has = [&](std::initializer_list<unsigned char> signature) {
        return bytes.size() >= signature.size() &&
            std::equal(signature.begin(), signature.end(), bytes.begin());
    };
    bool valid = true;
    switch (format) {
    case DocumentFormat::Png: valid = has({0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A}); break;
    case DocumentFormat::Jpeg: valid = has({0xFF, 0xD8, 0xFF}); break;
    case DocumentFormat::Pdf: valid = bytes.size() >= 5 && readText(bytes).compare(0, 5, "%PDF-") == 0; break;
    case DocumentFormat::Docx: valid = has({'P', 'K', 0x03, 0x04}) || has({'P', 'K', 0x05, 0x06}); break;
    case DocumentFormat::Txt: break;
    }
    if (!valid) throw std::invalid_argument("Содержимое не соответствует расширению файла");
}

// Извлекает обычный текст из минимального PDF. Полноценный OCR сканированных
// страниц выполняется через Options::pdfRasterizer и NeoML, см. комментарий выше.
std::string extractPdfText(const std::vector<unsigned char>& bytes) {
    const std::string data = readText(bytes);
    std::string result;
    std::regex literal(R"(\(([^()]*)\))");
    for (std::sregex_iterator it(data.begin(), data.end(), literal), end; it != end; ++it) {
        if (!result.empty()) result += ' ';
        result += (*it)[1].str();
    }
    return result;
}

// DOCX — zip-контейнер. Вызов unzip ограничен безопасно экранированным именем
// файла и используется только для чтения document.xml; для server deployment
// можно заменить этот маленький адаптер на minizip без изменения API.
std::string extractDocxText(const std::vector<unsigned char>& bytes) {
    const auto temporary = std::filesystem::temp_directory_path() /
        ("decan_document_" + std::to_string(std::hash<std::string>{}(readText(bytes))) + ".docx");
    {
        std::ofstream output(temporary, std::ios::binary);
        if (!output) throw std::runtime_error("Не удалось создать временный DOCX-файл");
        output.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    }
    std::string quoted = "'";
    const std::string path = temporary.string();
    for (char c : path) {
        if (c == '\'') quoted += "'\\''";
        else quoted += c;
    }
    quoted += "'";
    const std::string command = "unzip -p " + quoted + " word/document.xml 2>/dev/null";
    FILE* pipe = popen(command.c_str(), "r");
    if (!pipe) {
        std::filesystem::remove(temporary);
        throw std::runtime_error("Не удалось открыть DOCX-контейнер");
    }
    std::string xml;
    char buffer[4096];
    while (std::fgets(buffer, sizeof(buffer), pipe)) xml += buffer;
    const int status = pclose(pipe);
    std::filesystem::remove(temporary);
    if (status != 0 || xml.empty()) throw std::runtime_error("DOCX не содержит word/document.xml");
    std::string text;
    std::regex tag(R"(<[^>]+>)");
    std::sregex_token_iterator it(xml.begin(), xml.end(), tag, -1), end;
    for (; it != end; ++it) {
        if (!text.empty() && !std::isspace(static_cast<unsigned char>(text.back()))) text += ' ';
        text += it->str();
    }
    return text;
}

enum class OcrImageKind { Body, Header, Handwritten };

cv::Rect normalizedRect(const cv::Mat& image, double x, double y,
                       double width, double height) {
    const int left = std::clamp(static_cast<int>(std::lround(x * image.cols)), 0, image.cols);
    const int top = std::clamp(static_cast<int>(std::lround(y * image.rows)), 0, image.rows);
    const int right = std::clamp(static_cast<int>(std::lround((x + width) * image.cols)), left, image.cols);
    const int bottom = std::clamp(static_cast<int>(std::lround((y + height) * image.rows)), top, image.rows);
    return {left, top, right - left, bottom - top};
}

BoundingBox toBoundingBox(const cv::Rect& rect) {
    return {rect.x, rect.y, rect.width, rect.height};
}

cv::Mat prepareForOcr(const cv::Mat& source, const DocumentProcessor::Options& options,
                      OcrImageKind kind, double scale) {
    if (source.empty()) throw std::runtime_error("Пустое изображение документа");
    cv::Mat working = source;
    if (kind == OcrImageKind::Body && options.bodyBorderCropPixels > 0) {
        const int border = std::min({options.bodyBorderCropPixels,
                                     source.cols / 4, source.rows / 4});
        if (border > 0 && source.cols > border * 2 && source.rows > border * 2) {
            working = source(cv::Rect(border, border,
                                      source.cols - border * 2,
                                      source.rows - border * 2));
        }
    }

    cv::Mat gray, normalized, denoised, binary;
    if (working.channels() == 1) gray = working;
    else cv::cvtColor(working, gray, cv::COLOR_BGR2GRAY);

    // CLAHE поднимает локальный контраст в шапке и в слабых рукописных чернилах.
    auto clahe = cv::createCLAHE(kind == OcrImageKind::Handwritten ? 3.0 : 2.0,
                                 cv::Size(8, 8));
    clahe->apply(gray, normalized);
    if (kind == OcrImageKind::Handwritten) cv::GaussianBlur(normalized, denoised, {3, 3}, 0.0);
    else cv::medianBlur(normalized, denoised, 3);

    const int minDimension = std::min(denoised.rows, denoised.cols);
    if (minDimension < 3) {
        cv::threshold(denoised, binary, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    } else {
        int block = std::max(3, options.adaptiveBlockSize | 1);
        const int largestOddBlock = (minDimension % 2 == 0) ? minDimension - 1 : minDimension;
        block = std::min(block, largestOddBlock);
        cv::adaptiveThreshold(denoised, binary, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C,
                              cv::THRESH_BINARY, block, options.adaptiveConstant);
    }
    if (kind == OcrImageKind::Body) {
        const int morphologySize = std::clamp(options.bodyMorphologyKernelSize, 1, 7);
        const cv::Mat smallKernel = cv::getStructuringElement(
            cv::MORPH_RECT, {morphologySize, morphologySize});

        // Удаляет одиночные точки и короткие шумовые штрихи, не затрагивая
        // связные символы размером больше малого ядра.
        cv::morphologyEx(binary, binary, cv::MORPH_OPEN, smallKernel);

        if (options.removeBodyScanLines && binary.cols > 32 && binary.rows > 32) {
            cv::Mat foreground;
            cv::bitwise_not(binary, foreground);
            cv::Mat horizontal, vertical, lineMask;
            const int horizontalLength = std::max(15, foreground.cols / 25);
            const int verticalLength = std::max(15, foreground.rows / 25);
            cv::morphologyEx(foreground, horizontal, cv::MORPH_OPEN,
                cv::getStructuringElement(cv::MORPH_RECT, {horizontalLength, 1}));
            cv::morphologyEx(foreground, vertical, cv::MORPH_OPEN,
                cv::getStructuringElement(cv::MORPH_RECT, {1, verticalLength}));
            cv::bitwise_or(horizontal, vertical, lineMask);

            cv::Mat inverseLineMask, cleanedForeground;
            cv::bitwise_not(lineMask, inverseLineMask);
            cv::bitwise_and(foreground, inverseLineMask, cleanedForeground);
            cv::bitwise_not(cleanedForeground, binary);
        }
    }
    if (kind == OcrImageKind::Handwritten) {
        // Для рукописи сохраняем градации серого: тонкие штрихи часто теряются
        // при бинаризации, особенно на JPEG-сканах с голубыми чернилами.
        if (scale <= 1.0) return denoised;
        cv::Mat enlarged;
        cv::resize(denoised, enlarged, {}, scale, scale, cv::INTER_CUBIC);
        return enlarged;
    }
    if (scale <= 1.0) return binary;
    cv::Mat enlarged;
    cv::resize(binary, enlarged, {}, scale, scale, cv::INTER_CUBIC);
    return enlarged;
}

void whiten(cv::Mat& image, const cv::Rect& rect) {
    if (image.empty() || rect.width <= 0 || rect.height <= 0) return;
    const cv::Rect imageRect(0, 0, image.cols, image.rows);
    const cv::Rect safeRect = rect & imageRect;
    if (safeRect.width > 0 && safeRect.height > 0) {
        cv::rectangle(image, safeRect, cv::Scalar::all(255), cv::FILLED);
    }
}

} // namespace

const char* toString(DocumentFormat format) {
    switch (format) {
    case DocumentFormat::Png: return "png"; case DocumentFormat::Jpeg: return "jpeg";
    case DocumentFormat::Pdf: return "pdf"; case DocumentFormat::Docx: return "docx";
    case DocumentFormat::Txt: return "txt";
    }
    return "unknown";
}

const char* toString(VisualMark::Type type) {
    switch (type) {
    case VisualMark::Type::CoatOfArms: return "coat_of_arms";
    case VisualMark::Type::Seal: return "seal"; case VisualMark::Type::Stamp: return "stamp";
    case VisualMark::Type::Signature: return "signature";
    case VisualMark::Type::Unknown: return "unknown";
    }
    return "unknown";
}

const char* toString(HandwrittenField field) {
    switch (field) {
    case HandwrittenField::Date: return "date";
    case HandwrittenField::DocumentNumber: return "document_number";
    }
    return "unknown";
}

std::string DocumentAnalysis::toJson() const {
    std::ostringstream out;
    out << "{\"source\":\"" << jsonEscape(sourceName) << "\",\"format\":\""
        << toString(format) << "\",\"pages\":" << pageCount
        << ",\"text\":\"" << jsonEscape(text) << "\",\"visual_marks\":[";
    for (std::size_t i = 0; i < visualMarks.size(); ++i) {
        if (i) out << ',';
        const auto& mark = visualMarks[i];
        out << "{\"type\":\"" << toString(mark.type) << "\",\"confidence\":"
            << mark.confidence << ",\"x\":" << mark.bounds.x << ",\"y\":" << mark.bounds.y
            << ",\"width\":" << mark.bounds.width << ",\"height\":" << mark.bounds.height << '}';
    }
    out << "],\"regions\":[";
    for (std::size_t i = 0; i < regions.size(); ++i) {
        if (i) out << ',';
        const auto& region = regions[i];
        out << "{\"name\":\"" << jsonEscape(region.name) << "\",\"text\":\""
            << jsonEscape(region.text) << "\",\"raw_text\":\""
            << jsonEscape(region.rawText) << "\",\"confidence\":" << region.confidence
            << ",\"x\":" << region.bounds.x << ",\"y\":" << region.bounds.y
            << ",\"width\":" << region.bounds.width << ",\"height\":"
            << region.bounds.height << '}';
    }
    out << "],\"warnings\":[";
    for (std::size_t i = 0; i < warnings.size(); ++i) { if (i) out << ','; out << '"' << jsonEscape(warnings[i]) << '"'; }
    out << "]}";
    return out.str();
}

NeoMLAnalyzerAdapter::NeoMLAnalyzerAdapter(OcrFunction ocr, DetectorFunction detector)
    : NeoMLAnalyzerAdapter(std::move(ocr), HandwrittenOcrFunction{}, std::move(detector)) {}

NeoMLAnalyzerAdapter::NeoMLAnalyzerAdapter(OcrFunction printedOcr,
                                           HandwrittenOcrFunction handwrittenOcr,
                                           DetectorFunction detector)
    : ocr_(std::move(printedOcr)), handwrittenOcr_(std::move(handwrittenOcr)),
      detector_(std::move(detector)) {
    if (!ocr_ || !detector_) throw std::invalid_argument("NeoML OCR и detector функции обязательны");
}

NeoMLAnalyzerAdapter::NeoMLAnalyzerAdapter(NeoMLModelConfiguration models,
                                           OcrFunction printedFallback,
                                           HandwrittenOcrFunction handwrittenFallback,
                                           DetectorFunction detectorFallback)
    : ocr_(std::move(printedFallback)),
      handwrittenOcr_(std::move(handwrittenFallback)),
      detector_(std::move(detectorFallback)),
      onnxBackend_(std::make_unique<NeoMLOnnxBackend>(models)) {
    if (!ocr_ || !detector_) {
        throw std::invalid_argument("Для fallback нужны printed OCR и detector функции");
    }
    if (onnxBackend_) {
        for (const auto& message : onnxBackend_->diagnostics()) addDiagnostic(message);
    }
}

NeoMLAnalyzerAdapter::~NeoMLAnalyzerAdapter() = default;

NeoMLTextResult NeoMLAnalyzerAdapter::recognizeText(const cv::Mat& image, std::size_t page) {
    if (onnxBackend_) {
        try {
            return onnxBackend_->recognizeText(image, page);
        } catch (const std::exception& error) {
            addDiagnostic(std::string("NeoML printed OCR не отработал: ") + error.what() +
                          "; используется Tesseract fallback");
        }
    }
    return ocr_(image, page);
}
NeoMLTextResult NeoMLAnalyzerAdapter::recognizeHandwrittenText(const cv::Mat& image,
                                                               std::size_t page,
                                                               HandwrittenField field) {
    if (onnxBackend_) {
        try {
            return onnxBackend_->recognizeHandwritten(image, page, field);
        } catch (const std::exception& error) {
            addDiagnostic(std::string("NeoML handwriting OCR не отработал для ") +
                          toString(field) + ": " + error.what() + "; используется Tesseract fallback");
        }
    }
    return handwrittenOcr_ ? handwrittenOcr_(image, page, field) : ocr_(image, page);
}

std::vector<VisualMark> NeoMLAnalyzerAdapter::detectVisualMarks(const cv::Mat& image,
                                                                std::size_t page) {
    if (onnxBackend_) {
        try {
            return onnxBackend_->detectVisualMarks(image, page);
        } catch (const std::exception& error) {
            addDiagnostic(std::string("NeoML detector не отработал: ") + error.what() +
                          "; используется локальный detector fallback");
        }
    }
    return detector_(image, page);
}

void NeoMLAnalyzerAdapter::addDiagnostic(const std::string& message) const {
    if (std::find(diagnostics_.begin(), diagnostics_.end(), message) == diagnostics_.end()) {
        diagnostics_.push_back(message);
    }
}

std::vector<std::string> NeoMLAnalyzerAdapter::diagnostics() const {
    return diagnostics_;
}

DocumentProcessor::DocumentProcessor(std::shared_ptr<INeoMLAnalyzer> analyzer)
    : DocumentProcessor(std::move(analyzer), Options{}) {}

DocumentProcessor::DocumentProcessor(std::shared_ptr<INeoMLAnalyzer> analyzer, Options options)
    : analyzer_(std::move(analyzer)), options_(std::move(options)) {
    if (!analyzer_) throw std::invalid_argument("Для обработки нужен NeoML analyzer");
    if (options_.maxInputBytes == 0) throw std::invalid_argument("maxInputBytes не может быть равен нулю");
}

DocumentAnalysis DocumentProcessor::processPath(const std::string& path) const {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) throw std::runtime_error("Не удалось открыть файл: " + path);
    const auto size = input.tellg();
    if (size < 0 || static_cast<std::size_t>(size) > options_.maxInputBytes) throw std::length_error("Файл превышает допустимый размер");
    std::vector<unsigned char> bytes(static_cast<std::size_t>(size));
    input.seekg(0); input.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!input && !bytes.empty()) throw std::runtime_error("Ошибка чтения файла: " + path);
    return processBuffer(path, bytes);
}

DocumentAnalysis DocumentProcessor::processBuffer(const std::string& fileName, const std::vector<unsigned char>& bytes) const {
    if (bytes.empty()) throw std::invalid_argument("Передан пустой буфер");
    if (bytes.size() > options_.maxInputBytes) throw std::length_error("Буфер превышает допустимый размер");
    const DocumentFormat format = formatFromName(fileName);
    validateSignature(format, bytes);
    DocumentAnalysis result;
    result.sourceName = fileName;
    result.format = format;
    const auto appendAnalyzerDiagnostics = [&]() {
        for (const auto& warning : analyzer_->diagnostics()) {
            if (std::find(result.warnings.begin(), result.warnings.end(), warning) == result.warnings.end()) {
                result.warnings.push_back(warning);
            }
        }
    };
    if (format == DocumentFormat::Txt) {
        result.text = TextPostProcessor::correctPrintedText(readText(bytes));
        result.pageCount = 1;
        appendAnalyzerDiagnostics();
        return result;
    }
    if (format == DocumentFormat::Docx) {
        result.text = TextPostProcessor::correctPrintedText(extractDocxText(bytes));
        result.pageCount = 1;
        appendAnalyzerDiagnostics();
        return result;
    }
    if (format == DocumentFormat::Pdf && !options_.pdfRasterizer) {
        result.text = TextPostProcessor::correctPrintedText(extractPdfText(bytes));
        result.pageCount = 1;
        result.warnings.push_back("PDF не растеризован: подключите pdfRasterizer для OCR сканированных страниц");
        appendAnalyzerDiagnostics();
        return result;
    }
    std::vector<cv::Mat> pages;
    if (format == DocumentFormat::Pdf) pages = options_.pdfRasterizer(bytes);
    else pages.push_back(cv::imdecode(bytes, cv::IMREAD_COLOR));
    if (pages.empty()) throw std::runtime_error("Не удалось декодировать документ");
    result.pageCount = pages.size();
    for (std::size_t page = 0; page < pages.size(); ++page) {
        const cv::Mat& original = pages[page];
        if (original.empty()) throw std::runtime_error("Пустая страница документа");
        const cv::Rect headerRect(0, 0, original.cols,
                                  std::clamp(static_cast<int>(std::lround(original.rows * options_.headerFraction)),
                                             1, original.rows));
        const cv::Rect dateRect = normalizedRect(original, options_.handwrittenDateX,
            options_.handwrittenDateY, options_.handwrittenDateWidth, options_.handwrittenDateHeight);
        const cv::Rect numberRect = normalizedRect(original, options_.handwrittenNumberX,
            options_.handwrittenNumberY, options_.handwrittenNumberWidth, options_.handwrittenNumberHeight);

        auto appendResult = [&](const std::string& name, const cv::Rect& bounds,
                                std::string text, float confidence,
                                std::string rawText = {}) {
            result.regions.push_back({name, toBoundingBox(bounds), text,
                                      std::move(rawText), confidence});
            if (!text.empty()) {
                if (!result.text.empty()) result.text += '\n';
                result.text += text;
            }
        };

        // Печатный OCR страницы не должен видеть ни шапку, ни рукописные поля:
        // это устраняет дублирование и не даёт печатной модели читать почерк.
        cv::Mat body = original.clone();
        whiten(body, headerRect);
        whiten(body, dateRect);
        whiten(body, numberRect);
        const auto bodyOcr = analyzer_->recognizeText(
            prepareForOcr(body, options_, OcrImageKind::Body, options_.printedUpscale), page);

        if (options_.runHeaderPass) {
            cv::Mat header = original(headerRect).clone();
            whiten(header, {dateRect.x, dateRect.y, dateRect.width, dateRect.height});
            whiten(header, {numberRect.x, numberRect.y, numberRect.width, numberRect.height});
            const auto headerOcr = analyzer_->recognizeText(
                prepareForOcr(header, options_, OcrImageKind::Header, options_.headerUpscale), page);
            appendResult("header", headerRect,
                         TextPostProcessor::correctPrintedText(headerOcr.text), headerOcr.confidence,
                         headerOcr.text);
        }

        if (options_.runHandwrittenPass) {
            if (dateRect.width > 0 && dateRect.height > 0) {
                const auto dateOcr = analyzer_->recognizeHandwrittenText(
                    prepareForOcr(original(dateRect), options_, OcrImageKind::Handwritten,
                                  options_.handwritingUpscale), page, HandwrittenField::Date);
                const auto dateCorrection = TextPostProcessor::correctHandwrittenFieldDetailed(
                    dateOcr.text, HandwrittenField::Date);
                appendResult("handwritten_date", dateRect, dateCorrection.value,
                             dateOcr.confidence, dateCorrection.raw);
                if (!dateCorrection.accepted) {
                    result.warnings.push_back("Дата отклонена после OCR, raw=" + dateCorrection.raw);
                }
            } else {
                result.warnings.push_back("ROI рукописной даты имеет недопустимый размер");
            }

            if (numberRect.width > 0 && numberRect.height > 0) {
                const auto numberOcr = analyzer_->recognizeHandwrittenText(
                    prepareForOcr(original(numberRect), options_, OcrImageKind::Handwritten,
                                  options_.handwritingUpscale), page, HandwrittenField::DocumentNumber);
                const auto numberCorrection = TextPostProcessor::correctHandwrittenFieldDetailed(
                    numberOcr.text, HandwrittenField::DocumentNumber);
                appendResult("handwritten_document_number", numberRect,
                             numberCorrection.value, numberOcr.confidence,
                             numberCorrection.raw);
                if (!numberCorrection.accepted) {
                    result.warnings.push_back("Номер документа отклонён после OCR, raw=" + numberCorrection.raw);
                }
            } else {
                result.warnings.push_back("ROI рукописного номера имеет недопустимый размер");
            }
        }

        appendResult("body", {0, headerRect.height, original.cols,
                              original.rows - headerRect.height},
                     TextPostProcessor::correctPrintedText(bodyOcr.text), bodyOcr.confidence,
                     bodyOcr.text);

        auto marks = analyzer_->detectVisualMarks(pages[page], page);
        result.visualMarks.insert(result.visualMarks.end(), marks.begin(), marks.end());
    }
    result.text = TextPostProcessor::correctPrintedText(result.text);
    appendAnalyzerDiagnostics();
    return result;
}

} // namespace decan
