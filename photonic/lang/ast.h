// ast.h — AST node definitions for PhoLang
#ifndef AST_H
#define AST_H

// ─── AST Node Types ─────────────────────────────────────────────────

typedef enum {
    AST_PROGRAM,
    AST_LAYER_DECL,       // photon layer dense(...)
    AST_FORWARD_BLOCK,    // optical forward(x) { ... }
    AST_TRAIN_BLOCK,      // train with ... { ... }
    AST_INTERFERE_STMT,   // interfere x through weights
    AST_ACTIVATE_STMT,    // activate as sigmoid_optical
    AST_EMIT_STMT,        // emit output
    AST_IDENTIFIER,
    AST_NUMBER_LITERAL,
    AST_STRING_LITERAL,
    AST_PARAM_LIST,
    AST_CONFIG_PAIR,      // key: value
} ASTNodeType;

// ─── AST Node ───────────────────────────────────────────────────────

typedef struct ASTNode {
    ASTNodeType      type;
    const char      *value;        // string value (identifier, literal)
    double           num_value;    // numeric value
    struct ASTNode  *children;     // first child
    struct ASTNode  *next;         // sibling
    int              line;         // source line number
} ASTNode;

// Create a new AST node
ASTNode *ast_new(ASTNodeType type, const char *value, int line);

// Free an entire AST tree
void ast_free(ASTNode *node);

// Pretty-print AST for debugging
void ast_print(const ASTNode *node, int indent);

#endif // AST_H
