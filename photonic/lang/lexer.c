#include "lexer.h"
#include <string.h>
#include <ctype.h>

// Initialize lexer
Lexer lexer_init(const char *source) {
    Lexer lex = {
        .source  = source,
        .current = source,
        .line    = 1
    };
    return lex;
}

// Skip whitespace and comments
static void skip_whitespace(Lexer *lex) {
    for (;;) {
        char c = *lex->current;
        switch (c) {
            case ' ':
            case '\r':
            case '\t':
                lex->current++;
                break;
            case '\n':
                lex->line++;
                lex->current++;
                break;
            case '/':
                if (lex->current[1] == '/') {
                    // Single-line comment
                    while (*lex->current != '\n' && *lex->current != '\0') {
                        lex->current++;
                    }
                } else {
                    return;
                }
                break;
            default:
                return;
        }
    }
}

static int is_alpha(char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_';
}

static int is_digit(char c) {
    return c >= '0' && c <= '9';
}

// Helper to match identifier to keyword
static TokenType check_keyword(const char *start, int length, const char *keyword, TokenType type) {
    if ((int)strlen(keyword) == length && memcmp(start, keyword, length) == 0) {
        return type;
    }
    return TOK_IDENTIFIER;
}

static TokenType get_identifier_type(const char *start, int length) {
    #define CHECK(kw, type) { TokenType t = check_keyword(start, length, kw, type); if (t != TOK_IDENTIFIER) return t; }
    
    CHECK("photon", TOK_PHOTON);
    CHECK("network", TOK_NETWORK);
    CHECK("lens", TOK_LENS);
    CHECK("waveguide", TOK_LENS);
    CHECK("layer", TOK_LAYER);
    CHECK("mzi_mesh", TOK_LAYER);
    CHECK("readout", TOK_READOUT);
    CHECK("train", TOK_TRAIN);
    
    CHECK("optical", TOK_OPTICAL);
    CHECK("forward", TOK_FORWARD);
    CHECK("interfere", TOK_INTERFERE);
    CHECK("through", TOK_THROUGH);
    CHECK("activate", TOK_ACTIVATE);
    CHECK("as", TOK_AS);
    CHECK("emit", TOK_EMIT);
    CHECK("with", TOK_WITH);
    
    #undef CHECK
    return TOK_IDENTIFIER;
}

// Get next token
Token lexer_next(Lexer *lex) {
    skip_whitespace(lex);
    
    Token token;
    token.start = lex->current;
    token.line = lex->line;
    token.length = 0;
    
    if (*lex->current == '\0') {
        token.type = TOK_EOF;
        return token;
    }
    
    char c = *lex->current++;
    token.length = 1;
    
    // String Literals
    if (c == '"') {
        token.start = lex->current; // Do not include opening quote in string token value
        while (*lex->current != '"' && *lex->current != '\0') {
            if (*lex->current == '\\' && lex->current[1] != '\0') {
                lex->current += 2; // skip escaped character (e.g. \", \\, \n)
                continue;
            }
            if (*lex->current == '\n') {
                lex->line++;
            }
            lex->current++;
        }
        token.length = lex->current - token.start;
        if (*lex->current == '"') {
            lex->current++; // consume closing quote
        }
        token.type = TOK_STRING;
        return token;
    }
    
    // Single characters
    switch (c) {
        case '{': token.type = TOK_LBRACE; return token;
        case '}': token.type = TOK_RBRACE; return token;
        case '(': token.type = TOK_LPAREN; return token;
        case ')': token.type = TOK_RPAREN; return token;
        case '[': token.type = TOK_LBRACKET; return token;
        case ']': token.type = TOK_RBRACKET; return token;
        case ':': token.type = TOK_COLON; return token;
        case ',': token.type = TOK_COMMA; return token;
        case '.': token.type = TOK_DOT; return token;
        case ';': token.type = TOK_SEMICOLON; return token;
        case '=': token.type = TOK_EQUALS; return token;
        case '-': token.type = TOK_MINUS; return token;
    }
    
    // Identifiers & Keywords
    if (is_alpha(c)) {
        while (is_alpha(*lex->current) || is_digit(*lex->current)) {
            lex->current++;
        }
        token.length = lex->current - token.start;
        token.type = get_identifier_type(token.start, token.length);
        return token;
    }
    
    // Numbers (including float notation like 0.1, 1e-4)
    if (is_digit(c)) {
        while (is_digit(*lex->current)) {
            lex->current++;
        }
        // Fractional part
        if (*lex->current == '.' && is_digit(lex->current[1])) {
            lex->current++; // consume '.'
            while (is_digit(*lex->current)) {
                lex->current++;
            }
        }
        // Scientific notation (e.g. 1e-4)
        if ((*lex->current == 'e' || *lex->current == 'E')) {
            const char *peek = lex->current + 1;
            if (*peek == '+' || *peek == '-') peek++;
            if (is_digit(*peek)) {
                lex->current++; // consume 'e'
                if (*lex->current == '+' || *lex->current == '-') {
                    lex->current++; // consume sign
                }
                while (is_digit(*lex->current)) {
                    lex->current++;
                }
            }
        }
        token.length = lex->current - token.start;
        token.type = TOK_NUMBER;
        return token;
    }
    
    // Error
    token.type = TOK_ERROR;
    return token;
}

// Lookahead helper without consuming token
Token lexer_peek(Lexer *lex) {
    Lexer saved = *lex;
    Token t = lexer_next(lex);
    *lex = saved;
    return t;
}

// Convert TokenType to string representation
const char *token_type_to_string(TokenType type) {
    switch (type) {
        case TOK_PHOTON: return "PHOTON";
        case TOK_NETWORK: return "NETWORK";
        case TOK_LENS: return "LENS";
        case TOK_LAYER: return "LAYER";
        case TOK_READOUT: return "READOUT";
        case TOK_TRAIN: return "TRAIN";
        
        case TOK_OPTICAL: return "OPTICAL";
        case TOK_FORWARD: return "FORWARD";
        case TOK_INTERFERE: return "INTERFERE";
        case TOK_THROUGH: return "THROUGH";
        case TOK_ACTIVATE: return "ACTIVATE";
        case TOK_AS: return "AS";
        case TOK_EMIT: return "EMIT";
        case TOK_WITH: return "WITH";

        case TOK_IDENTIFIER: return "IDENTIFIER";
        case TOK_NUMBER: return "NUMBER";
        case TOK_STRING: return "STRING";
        
        case TOK_LBRACE: return "LBRACE";
        case TOK_RBRACE: return "RBRACE";
        case TOK_LPAREN: return "LPAREN";
        case TOK_RPAREN: return "RPAREN";
        case TOK_LBRACKET: return "LBRACKET";
        case TOK_RBRACKET: return "RBRACKET";
        case TOK_COLON: return "COLON";
        case TOK_COMMA: return "COMMA";
        case TOK_DOT: return "DOT";
        case TOK_SEMICOLON: return "SEMICOLON";
        case TOK_EQUALS: return "EQUALS";
        case TOK_MINUS: return "MINUS";
        
        case TOK_EOF: return "EOF";
        case TOK_ERROR: return "ERROR";
    }
    return "UNKNOWN";
}
