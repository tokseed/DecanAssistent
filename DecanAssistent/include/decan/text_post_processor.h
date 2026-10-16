#pragma once

#include <string>

namespace decan {

enum class HandwrittenField { Date, DocumentNumber };

struct HandwrittenCorrection {
    std::string value;
    std::string raw;
    bool accepted = false;
};

// Небольшой детерминированный слой коррекции для часто встречающихся ошибок
// OCR в русскоязычных официальных документах. Он не заменяет языковую модель:
// словарь следует расширять по реальным документам проекта.
class TextPostProcessor final {
public:
    static std::string correctPrintedText(const std::string& text);
    static HandwrittenCorrection correctHandwrittenFieldDetailed(
        const std::string& text, HandwrittenField field);
    static std::string correctHandwrittenField(const std::string& text,
                                               HandwrittenField field);
};

} // namespace decan
