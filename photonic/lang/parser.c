#include "ast.h"
#include "lexer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ─── Parser State ───────────────────────────────────────────────────
typedef struct {
    Lexer  lexer;
    Token  current;
    Token  peek;
    int    had_error;
} Parser;

// Helper to advance the parser state
static void parser_advance(Parser *parser) {
    parser->current = parser->peek;
    parser->peek = lexer_next(&parser->lexer);
}

// Check current token type
static int parser_check(const Parser *parser, TokenType type) {
    return parser->current.type == type;
}

// Consume current token, raising an error if it doesn't match
static void parser_consume(Parser *parser, TokenType type, const char *err_msg) {
    if (parser->current.type == type) {
        parser_advance(parser);
        return;
    }
    fprintf(stderr, "[Parser Error] Line %d: %s (Got '%s')\n", 
            parser->current.line, err_msg, token_type_to_string(parser->current.type));
    parser->had_error = 1;
}

// Forward declarations of parsing functions
static ASTNode *parse_statement(Parser *parser);
static ASTNode *parse_program(Parser *parser);
static ASTNode *parse_lens(Parser *parser);
static ASTNode *parse_layer(Parser *parser);
static ASTNode *parse_readout(Parser *parser);
static ASTNode *parse_train(Parser *parser);

// Parse entire PhoLang source code into AST
ASTNode *parser_parse(const char *source) {
    Parser parser;
    parser.lexer = lexer_init(source);
    parser.had_error = 0;
    
    // Initialize current and peek tokens
    parser.current = lexer_next(&parser.lexer);
    parser.peek = lexer_next(&parser.lexer);
    
    ASTNode *ast = parse_program(&parser);
    
    if (parser.had_error) {
        if (ast) {
            ast_free(ast);
            ast = NULL;
        }
    }
    
    return ast;
}

// Parse photon network definition
static ASTNode *parse_program(Parser *parser) {
    parser_consume(parser, TOK_PHOTON, "Expected 'photon'");
    parser_consume(parser, TOK_NETWORK, "Expected 'network'");
    
    Token name_tok = parser->current;
    parser_consume(parser, TOK_IDENTIFIER, "Expected network name identifier");
    
    char name[256];
    int len = name_tok.length < 255 ? name_tok.length : 255;
    memcpy(name, name_tok.start, len);
    name[len] = '\0';
    
    ASTNode *root = ast_new(AST_PROGRAM, name, name_tok.line);
    if (!root) return NULL;
    
    parser_consume(parser, TOK_LBRACE, "Expected '{' to start network block");
    
    ASTNode *tail = NULL;
    while (!parser_check(parser, TOK_RBRACE) && !parser_check(parser, TOK_EOF) && !parser->had_error) {
        ASTNode *stmt = parse_statement(parser);
        if (stmt) {
            if (!root->children) {
                root->children = stmt;
            } else {
                tail->next = stmt;
            }
            tail = stmt;
        }
    }
    
    parser_consume(parser, TOK_RBRACE, "Expected '}' to end network block");
    return root;
}

// Parse a single block statement inside network
static ASTNode *parse_statement(Parser *parser) {
    if (parser_check(parser, TOK_LENS)) {
        return parse_lens(parser);
    } else if (parser_check(parser, TOK_LAYER)) {
        return parse_layer(parser);
    } else if (parser_check(parser, TOK_READOUT)) {
        return parse_readout(parser);
    } else if (parser_check(parser, TOK_TRAIN)) {
        return parse_train(parser);
    }
    
    fprintf(stderr, "[Parser Error] Line %d: Unexpected token '%s'\n", 
            parser->current.line, token_type_to_string(parser->current.type));
    parser->had_error = 1;
    parser_advance(parser);
    return NULL;
}

// Parse lens pool(28, 4)
static ASTNode *parse_lens(Parser *parser) {
    Token lens_tok = parser->current;
    parser_consume(parser, TOK_LENS, "Expected 'lens'");
    
    Token type_tok = parser->current;
    parser_consume(parser, TOK_IDENTIFIER, "Expected lens type (e.g. pool)");
    
    char type_name[256];
    int len = type_tok.length < 255 ? type_tok.length : 255;
    memcpy(type_name, type_tok.start, len);
    type_name[len] = '\0';
    
    ASTNode *lens_node = ast_new(AST_LAYER_DECL, type_name, lens_tok.line);
    if (!lens_node) return NULL;
    
    parser_consume(parser, TOK_LPAREN, "Expected '(' after lens type");
    
    ASTNode *tail = NULL;
    while (!parser_check(parser, TOK_RPAREN) && !parser_check(parser, TOK_EOF) && !parser->had_error) {
        Token num_tok = parser->current;
        parser_consume(parser, TOK_NUMBER, "Expected number parameter");
        
        char num_str[64];
        int nlen = num_tok.length < 63 ? num_tok.length : 63;
        memcpy(num_str, num_tok.start, nlen);
        num_str[nlen] = '\0';
        double val = atof(num_str);
        
        ASTNode *num_node = ast_new(AST_NUMBER_LITERAL, NULL, num_tok.line);
        if (num_node) {
            num_node->num_value = val;
            if (!lens_node->children) {
                lens_node->children = num_node;
            } else {
                tail->next = num_node;
            }
            tail = num_node;
        }
        
        if (parser_check(parser, TOK_COMMA)) {
            parser_advance(parser);
        }
    }
    
    parser_consume(parser, TOK_RPAREN, "Expected ')' to close lens definition");
    return lens_node;
}

// Parse layer unitary(16) kerr(0.1) gain(3.0)
static ASTNode *parse_layer(Parser *parser) {
    Token layer_tok = parser->current;
    parser_consume(parser, TOK_LAYER, "Expected 'layer'");
    
    Token type_tok = parser->current;
    parser_consume(parser, TOK_IDENTIFIER, "Expected layer type (e.g. unitary)");
    
    char type_name[256];
    int len = type_tok.length < 255 ? type_tok.length : 255;
    memcpy(type_name, type_tok.start, len);
    type_name[len] = '\0';
    
    ASTNode *layer_node = ast_new(AST_LAYER_DECL, type_name, layer_tok.line);
    if (!layer_node) return NULL;
    
    parser_consume(parser, TOK_LPAREN, "Expected '(' after layer type");
    
    Token dim_tok = parser->current;
    parser_consume(parser, TOK_NUMBER, "Expected layer dimension");
    
    char dim_str[64];
    int dlen = dim_tok.length < 63 ? dim_tok.length : 63;
    memcpy(dim_str, dim_tok.start, dlen);
    dim_str[dlen] = '\0';
    double dim_val = atof(dim_str);
    
    ASTNode *dim_node = ast_new(AST_NUMBER_LITERAL, NULL, dim_tok.line);
    if (dim_node) {
        dim_node->num_value = dim_val;
        layer_node->children = dim_node;
    }
    
    parser_consume(parser, TOK_RPAREN, "Expected ')' after layer dimension");
    
    ASTNode *tail = dim_node;
    
    // Optional configuration attributes like kerr(0.1) or gain(3.0)
    while (parser_check(parser, TOK_IDENTIFIER)) {
        Token attr_tok = parser->current;
        parser_advance(parser);
        
        char attr_name[256];
        int alen = attr_tok.length < 255 ? attr_tok.length : 255;
        memcpy(attr_name, attr_tok.start, alen);
        attr_name[alen] = '\0';
        
        parser_consume(parser, TOK_LPAREN, "Expected '(' after attribute name");
        
        Token val_tok = parser->current;
        parser_consume(parser, TOK_NUMBER, "Expected attribute value");
        
        char val_str[64];
        int vlen = val_tok.length < 63 ? val_tok.length : 63;
        memcpy(val_str, val_tok.start, vlen);
        val_str[vlen] = '\0';
        double val = atof(val_str);
        
        parser_consume(parser, TOK_RPAREN, "Expected ')' after attribute value");
        
        ASTNode *pair = ast_new(AST_CONFIG_PAIR, attr_name, attr_tok.line);
        if (pair) {
            pair->num_value = val;
            if (tail) {
                tail->next = pair;
            } else {
                layer_node->children = pair;
            }
            tail = pair;
        }
    }
    
    return layer_node;
}

// Parse readout softmax(10)
static ASTNode *parse_readout(Parser *parser) {
    Token readout_tok = parser->current;
    parser_consume(parser, TOK_READOUT, "Expected 'readout'");
    
    Token type_tok = parser->current;
    parser_consume(parser, TOK_IDENTIFIER, "Expected readout type (e.g. softmax)");
    
    char type_name[256];
    int len = type_tok.length < 255 ? type_tok.length : 255;
    memcpy(type_name, type_tok.start, len);
    type_name[len] = '\0';
    
    ASTNode *readout_node = ast_new(AST_LAYER_DECL, type_name, readout_tok.line);
    if (!readout_node) return NULL;
    
    parser_consume(parser, TOK_LPAREN, "Expected '(' after readout type");
    
    Token num_tok = parser->current;
    parser_consume(parser, TOK_NUMBER, "Expected readout size");
    
    char num_str[64];
    int nlen = num_tok.length < 63 ? num_tok.length : 63;
    memcpy(num_str, num_tok.start, nlen);
    num_str[nlen] = '\0';
    double val = atof(num_str);
    
    ASTNode *num_node = ast_new(AST_NUMBER_LITERAL, NULL, num_tok.line);
    if (num_node) {
        num_node->num_value = val;
        readout_node->children = num_node;
    }
    
    parser_consume(parser, TOK_RPAREN, "Expected ')' after readout size");
    return readout_node;
}

// Parse train riemannian(lr=0.5, epochs=5, batch_size=10)
static ASTNode *parse_train(Parser *parser) {
    Token train_tok = parser->current;
    parser_consume(parser, TOK_TRAIN, "Expected 'train'");
    
    Token opt_tok = parser->current;
    parser_consume(parser, TOK_IDENTIFIER, "Expected training optimizer type (e.g. riemannian)");
    
    char opt_name[256];
    int len = opt_tok.length < 255 ? opt_tok.length : 255;
    memcpy(opt_name, opt_tok.start, len);
    opt_name[len] = '\0';
    
    ASTNode *train_node = ast_new(AST_TRAIN_BLOCK, opt_name, train_tok.line);
    if (!train_node) return NULL;
    
    parser_consume(parser, TOK_LPAREN, "Expected '(' after optimizer type");
    
    ASTNode *tail = NULL;
    while (!parser_check(parser, TOK_RPAREN) && !parser_check(parser, TOK_EOF) && !parser->had_error) {
        Token key_tok = parser->current;
        parser_consume(parser, TOK_IDENTIFIER, "Expected config key identifier");
        
        char key[256];
        int klen = key_tok.length < 255 ? key_tok.length : 255;
        memcpy(key, key_tok.start, klen);
        key[klen] = '\0';
        
        parser_consume(parser, TOK_EQUALS, "Expected '=' assignment operator");
        
        Token val_tok = parser->current;
        parser_consume(parser, TOK_NUMBER, "Expected config numeric value");
        
        char val_str[64];
        int vlen = val_tok.length < 63 ? val_tok.length : 63;
        memcpy(val_str, val_tok.start, vlen);
        val_str[vlen] = '\0';
        double val = atof(val_str);
        
        ASTNode *pair = ast_new(AST_CONFIG_PAIR, key, key_tok.line);
        if (pair) {
            pair->num_value = val;
            if (!train_node->children) {
                train_node->children = pair;
            } else {
                tail->next = pair;
            }
            tail = pair;
        }
        
        if (parser_check(parser, TOK_COMMA)) {
            parser_advance(parser);
        }
    }
    
    parser_consume(parser, TOK_RPAREN, "Expected ')' to close training configuration");
    return train_node;
}
