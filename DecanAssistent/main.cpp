#include "include/decan/document_processor.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifndef DECAN_MODELS_DIR
#define DECAN_MODELS_DIR "models"
#endif

namespace {

std::string shellQuote(const std::string& value) {
    std::string result = "'";
    for (const char character : value) {
        if (character == '\'') result += "'\\''";
        else result += character;
    }
    return result + "'";
}

std::string environmentValue(const char* name, const std::string& fallback) {
    const char* value = std::getenv(name);
    return value == nullptr || *value == '\0' ? fallback : std::string(value);
}

std::string modelPath(const std::filesystem::path& modelDirectory,
                      const char* environmentName, const char* fileName) {
    return environmentValue(environmentName, (modelDirectory / fileName).string());
}

// ONNX CTC-модели используют алфавит как список Unicode-кодовых точек, а не
// как массив отдельных байт. Это важно для кириллицы: один русский символ
// занимает два байта в UTF-8.
void appendUtf8Alphabet(const std::string& encoded, std::vector<std::string>& alphabet) {
    for (std::size_t position = 0; position < encoded.size();) {
        const unsigned char first = static_cast<unsigned char>(encoded[position]);
        std::size_t length = 1;
        if ((first & 0x80U) == 0) length = 1;
        else if ((first & 0xE0U) == 0xC0U) length = 2;
        else if ((first & 0xF0U) == 0xE0U) length = 3;
        else if ((first & 0xF8U) == 0xF0U) length = 4;

        if (position + length > encoded.size()) length = 1;
        bool valid = true;
        for (std::size_t offset = 1; valid && offset < length; ++offset) {
            const unsigned char continuation =
                static_cast<unsigned char>(encoded[position + offset]);
            valid = (continuation & 0xC0U) == 0x80U;
        }
        if (!valid) length = 1;
        alphabet.emplace_back(encoded.substr(position, length));
        position += length;
    }
}

decan::NeoMLTextResult runTesseract(const cv::Mat& image, std::size_t page,
                                    const std::string& suffix, const std::string& options) {
    const auto temporary = std::filesystem::temp_directory_path() /
        ("decan_ocr_" + std::to_string(page) + "_" + suffix + ".png");
    if (!cv::imwrite(temporary.string(), image)) {
        throw std::runtime_error("Не удалось подготовить страницу для OCR");
    }

    // Xcode запускает приложение с урезанным PATH, поэтому используем
    // абсолютный путь Homebrew вместо команды "tesseract".
    const std::string command = "/opt/homebrew/bin/tesseract " + shellQuote(temporary.string()) +
        " stdout -l rus+eng " + options + " 2>/dev/null";
    FILE* pipe = popen(command.c_str(), "r");
    if (!pipe) {
        std::filesystem::remove(temporary);
        throw std::runtime_error("Не удалось запустить Tesseract OCR");
    }
    std::string text;
    char buffer[4096];
    while (std::fgets(buffer, sizeof(buffer), pipe) != nullptr) text += buffer;
    const int status = pclose(pipe);
    std::filesystem::remove(temporary);
    if (status != 0) throw std::runtime_error("Tesseract не смог обработать изображение");
    return {text, text.empty() ? 0.0F : 0.75F};
}

// Печатный и рукописный проходы намеренно используют разные режимы Tesseract.
// Эти функции остаются fallback-реализацией, когда соответствующие NeoML-модели
// отсутствуют или не совпадают по входному/выходному формату.
decan::NeoMLTextResult localPrintedOcr(const cv::Mat& image, std::size_t page) {
    return runTesseract(image, page, "printed", "--oem 1 --psm 6");
}

decan::NeoMLTextResult localHandwrittenOcr(const cv::Mat& image, std::size_t page,
                                           decan::HandwrittenField field) {
    const std::string suffix = field == decan::HandwrittenField::Date ? "date" : "number";
    // Не ограничиваем whitelist слишком рано: на рукописном поле Tesseract
    // иногда отдаёт разделители или похожие символы, которые затем безопасно
    // нормализуются TextPostProcessor.
    return runTesseract(image, page, suffix, "--oem 1 --psm 7");
}

std::vector<decan::VisualMark> localVisualDetector(const cv::Mat& image, std::size_t) {
    std::vector<decan::VisualMark> marks;
    if (image.empty()) return marks;

    // Дешёвый fallback для герба в верхней части бланка. Это не заменяет
    // NeoML detector: результат имеет пониженную confidence и помечается как
    // fallback в основном приложении.
    const int headerHeight = std::max(1, static_cast<int>(image.rows * 0.18));
    const cv::Rect headerRect(0, 0, image.cols, headerHeight);
    cv::Mat gray, thresholded;
    if (image.channels() == 1) gray = image(headerRect);
    else cv::cvtColor(image(headerRect), gray, cv::COLOR_BGR2GRAY);
    cv::threshold(gray, thresholded, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);
    cv::morphologyEx(thresholded, thresholded, cv::MORPH_CLOSE,
                     cv::getStructuringElement(cv::MORPH_RECT, {5, 5}));

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(thresholded, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    double bestScore = 0.0;
    cv::Rect best;
    for (const auto& contour : contours) {
        if (contour.empty()) continue;
        int minX = std::numeric_limits<int>::max();
        int minY = std::numeric_limits<int>::max();
        int maxX = 0;
        int maxY = 0;
        for (const cv::Point& point : contour) {
            minX = std::min(minX, point.x);
            minY = std::min(minY, point.y);
            maxX = std::max(maxX, point.x);
            maxY = std::max(maxY, point.y);
        }
        const cv::Rect candidate(minX, minY, maxX - minX + 1, maxY - minY + 1);
        const double area = static_cast<double>(candidate.area());
        const double relativeArea = area / static_cast<double>(image.total());
        const double centerX = (candidate.x + candidate.width * 0.5) /
                               static_cast<double>(image.cols);
        const double height = candidate.height / static_cast<double>(image.rows);
        const double aspect = candidate.width / static_cast<double>(std::max(1, candidate.height));
        if (relativeArea < 0.0002 || relativeArea > 0.08 || centerX < 0.30 || centerX > 0.70 ||
            height < 0.015 || aspect < 0.20 || aspect > 3.5) continue;
        const double score = relativeArea * (1.0 - std::abs(centerX - 0.5));
        if (score > bestScore) {
            bestScore = score;
            best = {candidate.x, candidate.y, candidate.width, candidate.height};
        }
    }
    if (best.area() > 0) {
        best.y += headerRect.y;
        marks.push_back({decan::VisualMark::Type::CoatOfArms,
                         {best.x, best.y, best.width, best.height}, 0.45F});
    }
    return marks;
}

} // namespace

int main(int argc, char* argv[]) {
    // Изображение из задачи используется как вход по умолчанию. В дальнейшем
    // его можно заменить аргументом командной строки:
    // decan_core /путь/к/документу.jpg [/путь/к/models]
    const std::string inputPath = argc >= 2
        ? argv[1]
        : "/Users/pasavoronkov/Downloads/image.jpg";
    if (argc > 3) {
        std::cerr << "Использование: decan_core [путь-к-документу] [путь-к-models]\n";
        return 2;
    }

    try {
        const std::filesystem::path compiledModelDirectory(DECAN_MODELS_DIR);
        const std::filesystem::path defaultModelDirectory = compiledModelDirectory.is_absolute()
            ? compiledModelDirectory
            : std::filesystem::current_path() / compiledModelDirectory;
        const std::filesystem::path modelDirectory = argc == 3
            ? std::filesystem::path(argv[2])
            : std::filesystem::path(environmentValue(
                "DECAN_MODEL_DIR", defaultModelDirectory.string()));
        decan::NeoMLModelConfiguration models;
        models.printed.path = modelPath(modelDirectory,
            "DECAN_PRINTED_MODEL", "ocr_printed.onnx");
        models.handwriting.path = modelPath(modelDirectory,
            "DECAN_HANDWRITING_MODEL", "ocr_handwriting.onnx");
        models.sealStamp.path = modelPath(modelDirectory,
            "DECAN_SEAL_STAMP_MODEL", "seal_stamp_detector.onnx");
        models.coatOfArms.path = modelPath(modelDirectory,
            "DECAN_COAT_OF_ARMS_MODEL", "coat_of_arms_detector.onnx");

        // Каждая CTC-модель должна быть обучена с тем же порядком символов.
        // Алфавиты можно переопределить через переменные окружения.
        const std::string handwritingAlphabet = environmentValue(
            "DECAN_HANDWRITING_ALPHABET", "0123456789./№-");
        appendUtf8Alphabet(handwritingAlphabet, models.handwriting.alphabet);
        const std::string printedAlphabet = environmentValue(
            "DECAN_PRINTED_ALPHABET",
            " абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
            "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
            "0123456789.,:;!?-№()/%«»\"");
        appendUtf8Alphabet(printedAlphabet, models.printed.alphabet);
        models.sealStamp.classTypes = {decan::VisualMark::Type::Seal,
                                       decan::VisualMark::Type::Stamp};
        models.coatOfArms.forcedType = decan::VisualMark::Type::CoatOfArms;
        models.sealStamp.inputResolution = cv::Size(640, 640);
        models.sealStamp.letterbox = true;
        models.sealStamp.confidenceThreshold = 0.45F;
        models.sealStamp.nmsIouThreshold = 0.40F;
        models.coatOfArms.inputResolution = cv::Size(640, 640);
        models.coatOfArms.letterbox = true;
        models.coatOfArms.confidenceThreshold = 0.45F;
        models.coatOfArms.nmsIouThreshold = 0.40F;

        auto analyzer = std::make_shared<decan::NeoMLAnalyzerAdapter>(
            std::move(models), localPrintedOcr, localHandwrittenOcr, localVisualDetector);

        decan::DocumentProcessor processor(analyzer);
        auto analysis = processor.processPath(inputPath);
        std::cout << analysis.toJson() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Ошибка обработки: " << error.what() << '\n';
        return 1;
    }
}
