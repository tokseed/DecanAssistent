#pragma once

#include "../include/decan/document_processor.h"

#include <memory>
#include <string>
#include <vector>

namespace decan {

// Внутренний интерфейс backend-а. Заголовок не включает заголовки NeoML, что
// позволяет собирать сам модуль и Xcode-цель даже в конфигурации без vendored
// NeoML. Реализация подключается CMake-макросом DECAN_WITH_NEOML.
class NeoMLOnnxBackend {
public:
    explicit NeoMLOnnxBackend(const NeoMLModelConfiguration& configuration);
    ~NeoMLOnnxBackend();

    NeoMLTextResult recognizeText(const cv::Mat& image, std::size_t page);
    NeoMLTextResult recognizeHandwritten(const cv::Mat& image,
                                         std::size_t page,
                                         HandwrittenField field);
    std::vector<VisualMark> detectVisualMarks(const cv::Mat& image,
                                              std::size_t page);
    std::vector<std::string> diagnostics() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace decan
