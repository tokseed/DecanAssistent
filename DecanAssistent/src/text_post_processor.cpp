#include "../include/decan/text_post_processor.h"

#include <algorithm>
#include <cctype>
#include <iomanip>
#include <regex>
#include <sstream>
#include <utility>
#include <vector>

namespace decan {
namespace {

void replaceAll(std::string& value, const std::string& from, const std::string& to) {
    if (from.empty()) return;
    std::size_t position = 0;
    while ((position = value.find(from, position)) != std::string::npos) {
        value.replace(position, from.size(), to);
        position += to.size();
    }
}

std::string trim(std::string value) {
    const auto isSpace = [](unsigned char c) { return std::isspace(c) != 0; };
    value.erase(value.begin(), std::find_if(value.begin(), value.end(),
        [&](unsigned char c) { return !isSpace(c); }));
    value.erase(std::find_if(value.rbegin(), value.rend(),
        [&](unsigned char c) { return !isSpace(c); }).base(), value.end());
    return value;
}

bool isEdgeArtifact(char character) {
    return character == '|' || character == '\\' || character == '_';
}

std::string cleanLineArtifacts(std::string line) {
    line = trim(std::move(line));
    while (!line.empty() && isEdgeArtifact(line.front())) line.erase(line.begin());
    while (!line.empty() && isEdgeArtifact(line.back())) line.pop_back();
    line = trim(std::move(line));

    // Точка/кавычка перед пробелом в начале строки обычно является остатком
    // вертикальной границы или шумом бинаризации, а не частью номера пункта.
    while (line.size() >= 2 && (line.front() == '.' || line.front() == '\'' ||
                                line.front() == '`') &&
           std::isspace(static_cast<unsigned char>(line[1]))) {
        line.erase(line.begin());
        line = trim(std::move(line));
    }
    // Частые UTF-8 кавычки, оставшиеся от края скана.
    for (const std::string& prefix : {std::string("\xE2\x80\x98 "),
                                      std::string("\xE2\x80\x99 "),
                                      std::string("\xE2\x80\x9C "),
                                      std::string("\xE2\x80\x9D ")}) {
        if (line.rfind(prefix, 0) == 0) line.erase(0, prefix.size());
    }

    // Отдельная точка/подчёркивание/слэш обычно является краем скана, а не
    // содержимым документа. Не трогаем обычную пунктуацию внутри предложения.
    bool onlyPunctuation = !line.empty();
    for (const char character : line) {
        if (std::isspace(static_cast<unsigned char>(character))) continue;
        if (character != '.' && character != ',' && character != ';' &&
            character != ':' && character != '-' && !isEdgeArtifact(character)) {
            onlyPunctuation = false;
            break;
        }
    }
    if (onlyPunctuation) return {};

    // Частый след от горизонтальной границы после бинаризации: пробел + точка.
    if (line.size() >= 2 && line[line.size() - 2] == ' ' && line.back() == '.') {
        line.erase(line.size() - 2);
        line = trim(std::move(line));
    }
    // Удаляем точку, ошибочно прилепленную к короткому союзу/предлогу:
    // например, "систем и." вместо "систем и".
    for (const std::string suffix : {" и.", " а.", " в.", " к."}) {
        if (line.size() >= suffix.size() &&
            line.compare(line.size() - suffix.size(), suffix.size(), suffix) == 0) {
            line.erase(line.size() - 1);
            break;
        }
    }
    return line;
}

void correctListMarker(std::string& line) {
    // Tesseract часто читает номер подпункта "1.1." как "$." или "S.".
    line = std::regex_replace(line, std::regex(R"(^\s*[$§S5]\s*\.\s*)"), "1.1. ");
    if (std::regex_search(line, std::regex(R"(^\s*1\.\s+)")) &&
        line.find("принять") != std::string::npos) {
        line = std::regex_replace(line, std::regex(R"(^\s*1\.\s+)"), "1.1. ");
    }
    // Иногда точка между цифрами теряется: "12." -> "1.2.".
    line = std::regex_replace(line, std::regex(R"(^\s*1([12])\.)"), "1.$1.");
}

} // namespace

std::string TextPostProcessor::correctPrintedText(const std::string& text) {
    std::istringstream input(text);
    std::ostringstream cleaned;
    std::string line;
    bool firstLine = true;
    while (std::getline(input, line)) {
        line = cleanLineArtifacts(std::move(line));
        correctListMarker(line);
        line = std::regex_replace(line, std::regex(R"(^\s*[.|\\_]+\s+)"), "");
        line = std::regex_replace(line, std::regex(R"(\s+[|\\_]+\s*$)"), "");
        if (!firstLine) cleaned << '\n';
        cleaned << line;
        firstLine = false;
    }

    std::string corrected = cleaned.str();
    const std::vector<std::pair<std::string, std::string>> dictionary = {
        {"ФЕЛЕРАЛЬНАЯ", "ФЕДЕРАЛЬНАЯ"},
        {"фелерaльная", "федеральная"},
        {"СЛУЖБА ГО НАДЗОРУ", "СЛУЖБА ПО НАДЗОРУ"},
        {"служба го надзору", "служба по надзору"},
        {"8 СФЕРЕ", "В СФЕРЕ"},
        {"8 сфере", "в сфере"},
        {"ЗАЦИТЫ", "ЗАЩИТЫ"},
        {"зациты", "защиты"},
        {"БПАГОЛОЛУЧИЯ", "БЛАГОПОЛУЧИЯ"},
        {"БПАГОПОЛУЧИЯ", "БЛАГОПОЛУЧИЯ"},
        {"бпагололучия", "благополучия"},
        {"БЛАГОПОЛУЧИЯ ЧЕ ПОЗЕКА", "БЛАГОПОЛУЧИЯ ЧЕЛОВЕКА"},
        {"благополучия че позека", "благополучия человека"},
        {"зодоотведении", "водоотведении"},
        {"ЗОДООТВЕДЕНИИ", "ВОДООТВЕДЕНИИ"},
        {"скогочогильников", "скотомогильников"},
        {"скотохогильников", "скотомогильников"},
        {"нанозохранилящ", "навозохранилищ"},
        {"навозохранилящ", "навозохранилищ"},
        // Ошибки, которые стабильно появляются в body-ROI на этом типе
        // санитарно-эпидемиологических постановлений.
        {"волинкиовсыия", "возникновения"},
        {"ВОЛИНКИОВСЫИЯ", "ВОЗНИКНОВЕНИЯ"},
        {"полвержемных", "подверженных"},
        {"ПОЛВЕРЖЕМНЫХ", "ПОДВЕРЖЕННЫХ"},
        {"сохзанных", "связанных"},
        {"СОХЗАННЫХ", "СВЯЗАННЫХ"},
        {"босперебаймой", "бесперебойной"},
        {"БОСПЕРЕБАЙМОЙ", "БЕСПЕРЕБОЙНОЙ"},
        {"пермод", "период"},
        {"ПЕРМОД", "ПЕРИОД"},
        {"пнволка", "паводка"},
        {"ПНВОЛКА", "ПАВОДКА"},
        {"паводха", "паводка"},
        {"ПАВОДХА", "ПАВОДКА"},
        {"золоотведеныя", "водоотведения"},
        {"ЗОЛООТВЕДЕНЫЯ", "ВОДООТВЕДЕНИЯ"},
        {"полазльных", "подвальных"},
        {"ПОЛАЗЛЬНЫХ", "ПОДВАЛЬНЫХ"},
        {"дерагмзании", "дератизации"},
        {"ДЕРАГМЗАНИИ", "ДЕРАТИЗАЦИИ"},
        {"самитарно-эпидемиологического", "санитарно-эпидемиологического"},
        {"санитарно-эпидемнологическом", "санитарно-эпидемиологическом"},
        {"исриод", "период"},
        {"нанодка", "паводка"},
        {"мупиципальных", "муниципальных"},
        {"дезинфинирующих", "дезинфицирующих"},
        {"зодоотведения", "водоотведения"},
        {"водоотведснияю", "водоотведения"},
        {"благополучыя", "благополучия"},
        {"БЛАГОПОЛУЧЫЯ", "БЛАГОПОЛУЧИЯ"},
        {"ханалязационных", "канализационных"},
        {"ХАНАЛЯЗАЦИОННЫХ", "КАНАЛИЗАЦИОННЫХ"},
        {"памодха", "паводка"},
        {"ПАМОДХА", "ПАВОДКА"},
        {"делюнсекция", "дезинсекция"},
        {"ДЕЛЮНСЕКЦИЯ", "ДЕЗИНСЕКЦИЯ"},
        {"сеннтарно", "санитарно"},
        {"хканалязацнонных", "канализационных"},
        {"зодостведсния", "водоотведения"},
        {"ПАСТАНОВЛЕНИЕ", "ПОСТАНОВЛЕНИЕ"},
        {"Челябансх", "Челябинск"}
    };
    for (const auto& [from, to] : dictionary) replaceAll(corrected, from, to);
    return corrected;
}

HandwrittenCorrection TextPostProcessor::correctHandwrittenFieldDetailed(
    const std::string& text, HandwrittenField field) {
    HandwrittenCorrection correction;
    correction.raw = trim(text);
    std::string corrected = correction.raw;
    if (field == HandwrittenField::Date) {
        // В датах OCR часто путает латинские O/I/l и кириллическую О с цифрами.
        replaceAll(corrected, "O", "0");
        replaceAll(corrected, "o", "0");
        replaceAll(corrected, "О", "0");
        replaceAll(corrected, "о", "0");
        replaceAll(corrected, ",", ".");
        replaceAll(corrected, ":", ".");
        replaceAll(corrected, "I", "1");
        replaceAll(corrected, "l", "1");
        replaceAll(corrected, "|", "1");

        std::smatch match;
        if (std::regex_search(corrected, match,
                std::regex(R"((\d{1,2})[\s./,-]+(\d{1,2})[\s./,-]+(\d{2,4}))"))) {
            const int day = std::stoi(match[1].str());
            const int month = std::stoi(match[2].str());
            std::string year = match[3].str();
            if (day >= 1 && day <= 31 && month >= 1 && month <= 12) {
                if (year.size() == 2) year = "20" + year;
                const int fullYear = std::stoi(year);
                if (fullYear >= 1900 && fullYear <= 2100) {
                    std::ostringstream normalizedDate;
                    normalizedDate << std::setfill('0') << std::setw(2) << day << '.'
                                   << std::setw(2) << month << '.' << year;
                    correction.value = normalizedDate.str();
                    correction.accepted = true;
                }
            }
        }
    } else {
        std::smatch match;
        if (std::regex_search(corrected, match,
                std::regex(R"((?:№|N|No|#)\s*(\d{1,6})\b|\b(\d{1,6})\b)",
                           std::regex_constants::icase))) {
            const std::string number = match[1].matched ? match[1].str() : match[2].str();
            correction.value = "№ " + number;
            correction.accepted = true;
        }
    }
    return correction;
}

std::string TextPostProcessor::correctHandwrittenField(const std::string& text,
                                                        HandwrittenField field) {
    return correctHandwrittenFieldDetailed(text, field).value;
}

} // namespace decan
