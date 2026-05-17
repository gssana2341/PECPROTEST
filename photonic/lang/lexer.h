#ifndef LEXER_H
#define LEXER_H

// ─── Token Types ────────────────────────────────────────────────────
typedef enum {
    // Structural Keywords
    TOK_PHOTON, TOK_NETWORK, TOK_LENS, TOK_LAYER, TOK_READOUT, TOK_TRAIN,
    
    // Original abstract AST keywords
    TOK_OPTICAL, TOK_FORWARD, TOK_INTERFERE, TOK_THROUGH,
    TOK_ACTIVATE, TOK_AS, TOK_EMIT, TOK_WITH,

    // Literals & identifiers
    TOK_IDENTIFIER, TOK_NUMBER, TOK_STRING,

    // Delimiters & Operators
    TOK_LBRACE, TOK_RBRACE, TOK_LPAREN, TOK_RPAREN,
    TOK_LBRACKET, TOK_RBRACKET,
    TOK_COLON, TOK_COMMA, TOK_DOT, TOK_SEMICOLON,
    TOK_EQUALS,
    TOK_MINUS,

    // Special
    TOK_EOF, TOK_ERROR
} TokenType;

typedef struct {
    TokenType   type;
    const char *start;
    int         length;
    int         line;
} Token;

// ─── Lexer State ────────────────────────────────────────────────────
typedef struct {
    const char *source;
    const char *current;
    int         line;
} Lexer;

// Initialize lexer with source code
Lexer lexer_init(const char *source);

// Get next token
Token lexer_next(Lexer *lex);

// Look ahead at next token without consuming it
Token lexer_peek(Lexer *lex);

// Helper to get token type as string for debugging
const char *token_type_to_string(TokenType type);

#endif // LEXER_H
