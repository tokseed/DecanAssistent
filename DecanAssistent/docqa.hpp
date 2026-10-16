#pragma once
#include <opencv2/imgcodecs.hpp>
#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <regex>
#include <string>
#include <utility>
#include <vector>
#include <unistd.h>

namespace dq {
using std::string;
inline constexpr size_t NP = string::npos;

struct Box { int x1 = 0, y1 = 0, x2 = 0, y2 = 0; };
struct Item { string text; Box b; string kind; };
struct DocModel { int w = 0, h = 0; std::vector<Item> items; };

inline string MODEL = "qwen2.5:7b";
inline const string SYS = "Ты — ассистент по анализу документов. Отвечай только по приведённому тексту документа, на русском. Строго соблюдай формат ответа, указанный в вопросе. Больше ничего не пиши.";

inline string trim(string s) {
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back()))) s.pop_back();
    size_t i = 0;
    while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i]))) i++;
    return s.substr(i);
}

inline string cut(const string& s, size_t n) {
    if (s.size() <= n) return s;
    while (n > 0 && (static_cast<unsigned char>(s[n]) & 0xC0) == 0x80) n--;
    return s.substr(0, n);
}

inline string readPath(int argc, char** argv) {
    string p;
    if (argc > 1) p = argv[1];
    else { std::cout << "Путь к документу: " << std::flush; std::getline(std::cin, p); }
    p = trim(p);
    if (p.size() > 1 && (p.front() == '\'' || p.front() == '"') && p.back() == p.front()) p = p.substr(1, p.size() - 2);
    string o;
    for (size_t i = 0; i < p.size(); i++) { if (p[i] == '\\' && i + 1 < p.size()) i++; o += p[i]; }
    return trim(o);
}

inline bool loadDocument(const string& path, cv::Mat& out) {
    out = cv::imread(path, cv::IMREAD_COLOR);
    if (out.empty()) { std::cerr << "Не удалось загрузить: " << path << "\n"; return false; }
    return true;
}

inline string jesc(const string& s) {
    string o; char b[8];
    for (unsigned char c : s) {
        switch (c) {
            case '"': o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n"; break;
            case '\r': o += "\\r"; break;
            case '\t': o += "\\t"; break;
            default: if (c < 0x20) { std::snprintf(b, sizeof b, "\\u%04x", c); o += b; } else o += static_cast<char>(c);
        }
    }
    return o;
}

inline void utf8(string& o, unsigned cp) {
    if (cp < 0x80) o += static_cast<char>(cp);
    else if (cp < 0x800) { o += static_cast<char>(0xC0 | (cp >> 6)); o += static_cast<char>(0x80 | (cp & 0x3F)); }
    else if (cp < 0x10000) { o += static_cast<char>(0xE0 | (cp >> 12)); o += static_cast<char>(0x80 | ((cp >> 6) & 0x3F)); o += static_cast<char>(0x80 | (cp & 0x3F)); }
    else { o += static_cast<char>(0xF0 | (cp >> 18)); o += static_cast<char>(0x80 | ((cp >> 12) & 0x3F)); o += static_cast<char>(0x80 | ((cp >> 6) & 0x3F)); o += static_cast<char>(0x80 | (cp & 0x3F)); }
}

inline string extractResponse(const string& j) {
    const string k = "\"response\":\"";
    size_t p = j.find(k);
    if (p == NP) return "";
    p += k.size();
    string o;
    try {
        while (p < j.size()) {
            char c = j[p];
            if (c == '"') break;
            if (c == '\\' && p + 1 < j.size()) {
                char e = j[++p];
                if (e == 'n') o += '\n';
                else if (e == 't') o += '\t';
                else if (e == 'r') o += '\r';
                else if (e == 'u' && p + 4 < j.size()) {
                    unsigned cp = static_cast<unsigned>(std::stoul(j.substr(p + 1, 4), nullptr, 16));
                    p += 4;
                    if (cp >= 0xD800 && cp < 0xDC00 && p + 6 < j.size() && j.compare(p + 1, 2, "\\u") == 0) {
                        unsigned lo = static_cast<unsigned>(std::stoul(j.substr(p + 3, 4), nullptr, 16));
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                        p += 6;
                    }
                    utf8(o, cp);
                } else o += e;
            } else o += c;
            p++;
        }
    } catch (...) {}
    return o;
}

inline string curlPost(const string& payload) {
    char t[] = "/tmp/dqXXXXXX";
    int fd = mkstemp(t);
    if (fd < 0) return "";
    close(fd);
    { std::ofstream f(t, std::ios::binary); f << payload; }
    string cmd = string("curl -s http://localhost:11434/api/generate -H 'Content-Type: application/json' --data-binary @") + t;
    string r;
    if (FILE* p = popen(cmd.c_str(), "r")) {
        char b[4096]; size_t n;
        while ((n = fread(b, 1, sizeof b, p)) > 0) r.append(b, n);
        pclose(p);
    }
    std::remove(t);
    return r;
}

inline string llm(const string& sys, const string& user) {
    string pl = "{\"model\":\"" + MODEL + "\",\"system\":\"" + jesc(sys) + "\",\"prompt\":\"" + jesc(user) +
                "\",\"stream\":false,\"options\":{\"temperature\":0,\"num_ctx\":8192}}";
    return extractResponse(curlPost(pl));
}

inline string lower(const string& s) {
    string o; o.reserve(s.size());
    for (size_t i = 0; i < s.size(); i++) {
        unsigned char c = static_cast<unsigned char>(s[i]);
        if (c == 0xD0 && i + 1 < s.size()) {
            unsigned char d = static_cast<unsigned char>(s[i + 1]);
            if (d >= 0x90 && d <= 0x9F) { o += static_cast<char>(0xD0); o += static_cast<char>(d + 0x20); i++; continue; }
            if (d >= 0xA0 && d <= 0xAF) { o += static_cast<char>(0xD1); o += static_cast<char>(d - 0x20); i++; continue; }
            if (d == 0x81) { o += static_cast<char>(0xD1); o += static_cast<char>(0x91); i++; continue; }
        }
        o += c < 0x80 ? static_cast<char>(std::tolower(c)) : static_cast<char>(c);
    }
    return o;
}

inline bool has(const string& s, std::initializer_list<const char*> w) {
    for (auto x : w) if (s.find(x) != NP) return true;
    return false;
}

inline string classify(const string& q0) {
    string q = lower(q0);
    if (has(q, {"координат", "bounding", "bbox", "ограничивающ"})) return "grounding";
    if (has(q, {"markdown", "markdawn", "маркдаун", "маркдавн", "оттекстов"})) return "parse";
    if (has(q, {"словар", "ключ"})) return "kie";
    if (has(q, {"где ", "покажи", "отыщи", "выдели", "найди на"})) return "grounding";
    if (has(q, {"распозна", "прочт", "прочит", "считай", "отсканир", "извлеки весь", "текст"})) return "ocr";
    return "vqa";
}

inline string fmt(const Box& b) {
    return "(" + std::to_string(b.x1) + ", " + std::to_string(b.y1) + ", " + std::to_string(b.x2) + ", " + std::to_string(b.y2) + ")";
}

inline long long area(const Box& b) { return static_cast<long long>(b.x2 - b.x1) * (b.y2 - b.y1); }

inline std::vector<const Item*> textItems(const DocModel& m) {
    bool L = false;
    for (auto& i : m.items) if (i.kind == "line") L = true;
    std::vector<const Item*> c;
    for (auto& i : m.items) if (i.kind == (L ? "line" : "word") && !i.text.empty()) c.push_back(&i);
    return c;
}

inline string readingText(const DocModel& m) {
    std::vector<Item> v;
    for (auto p : textItems(m)) v.push_back(*p);
    auto cy = [](const Item& a) { return (a.b.y1 + a.b.y2) / 2; };
    std::sort(v.begin(), v.end(), [&](const Item& a, const Item& b) { return cy(a) < cy(b); });
    string o;
    size_t i = 0;
    while (i < v.size()) {
        size_t j = i;
        int ref = cy(v[i]), th = std::max(6, (v[i].b.y2 - v[i].b.y1) * 6 / 10);
        while (j < v.size() && cy(v[j]) - ref <= th) j++;
        std::sort(v.begin() + i, v.begin() + j, [](const Item& a, const Item& b) { return a.b.x1 < b.b.x1; });
        for (size_t k = i; k < j; k++) { if (k > i) o += ' '; o += v[k].text; }
        o += '\n';
        i = j;
    }
    o = cut(o, 16000);
    for (auto& it : m.items)
        if (it.kind == "emblem" || it.kind == "seal")
            o += string("[") + (it.kind == "emblem" ? "герб" : "печать") + ": " + fmt(it.b) + "]\n";
    return o;
}

inline string quotedPhrase(const string& s) {
    size_t a = s.find("«");
    if (a != NP) { size_t b = s.find("»", a); if (b != NP) return s.substr(a + 2, b - a - 2); }
    a = s.find('"');
    if (a != NP) { size_t b = s.find('"', a + 1); if (b != NP) return s.substr(a + 1, b - a - 1); }
    return "";
}

inline int pickIndex(const std::vector<const Item*>& c, const string& q) {
    string l;
    for (size_t i = 0; i < c.size() && i < 200; i++) l += std::to_string(i) + ": " + c[i]->text + "\n";
    string a = llm(SYS, "Список фрагментов документа:\n" + l + "\nВопрос: " + q +
                        "\nВыбери номер фрагмента, который лучше всего отвечает на вопрос. Ответь только числом.");
    std::smatch mt;
    if (std::regex_search(a, mt, std::regex("\\d+"))) {
        size_t k = std::stoul(mt[0]);
        if (k < c.size() && k < 200) return static_cast<int>(k);
    }
    return -1;
}

inline string ground(const DocModel& m, const string& q) {
    string l = lower(q);
    string kind = has(l, {"герб"}) ? "emblem" : has(l, {"печат"}) ? "seal" : "";
    if (!kind.empty()) {
        string o;
        for (auto& i : m.items) if (i.kind == kind) { if (!o.empty()) o += "\n"; o += fmt(i.b); }
        return o.empty() ? "Не найдено на документе." : o;
    }
    auto c = textItems(m);
    string ph = quotedPhrase(l);
    if (!ph.empty()) {
        const Item* best = nullptr;
        for (auto p : c)
            if (lower(p->text).find(ph) != NP && (!best || area(p->b) < area(best->b))) best = p;
        if (best) return fmt(best->b);
    }
    int k = pickIndex(c, q);
    return k >= 0 ? fmt(c[k]->b) : "Не удалось определить положение.";
}

inline string post(const string& t, const string& q, string a) {
    a = trim(a);
    if (a.rfind("```", 0) == 0) {
        size_t e = a.find('\n');
        a = e == NP ? "" : a.substr(e + 1);
        size_t f = a.rfind("```");
        if (f != NP) a = a.substr(0, f);
        a = trim(a);
    }
    if (t == "kie") {
        size_t s = a.find('{'), e = a.rfind('}');
        if (s != NP && e != NP && e > s) {
            a = a.substr(s, e - s + 1);
            if (a.find('\'') == NP) for (auto& c : a) if (c == '"') c = '\'';
        }
    } else if (t == "vqa") {
        if (!a.empty() && a.size() < 120 && a.find('\n') == NP && lower(q).find("предложен") == NP && a.back() == '.') a.pop_back();
        if (a.size() >= 2 && a.front() == '"' && a.back() == '"') a = a.substr(1, a.size() - 2);
        if (a.size() >= 4 && a.compare(0, 2, "«") == 0 && a.compare(a.size() - 2, 2, "»") == 0) a = a.substr(2, a.size() - 4);
    }
    return a;
}

inline string ask(const DocModel& m, const string& q) {
    static std::deque<std::pair<string, string>> h;
    string t = classify(q), a;
    if (t == "grounding") a = ground(m, q);
    else {
        string hist;
        for (auto& e : h) hist += "Вопрос: " + e.first + "\nОтвет: " + cut(e.second, 300) + "\n";
        string pr = "Документ:\n" + readingText(m);
        if (!hist.empty()) pr += "\nИстория диалога:\n" + hist;
        pr += "\nВопрос: " + q;
        if (t == "kie") pr += "\nВерни только словарь Python вида {'ключ': ['значение']}.";
        if (t == "parse") pr += "\nОтвет оформи в Markdown.";
        a = post(t, q, llm(SYS, pr));
        if (a.empty()) a = "Нет ответа от Ollama: проверьте, запущен ли ollama serve и установлена ли модель " + MODEL + ".";
    }
    h.emplace_back(q, a);
    if (h.size() > 6) h.pop_front();
    return a;
}

inline void chat(const DocModel& m) {
    string q;
    while (true) {
        std::cout << "> " << std::flush;
        if (!std::getline(std::cin, q) || q == "exit") break;
        if (trim(q).empty()) continue;
        std::cout << ask(m, q) << "\n";
    }
}
}
