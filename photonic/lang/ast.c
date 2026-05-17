#include "ast.h"
#include "../core/memory.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Create a new AST node
ASTNode *ast_new(ASTNodeType type, const char *value, int line) {
    ASTNode *node = (ASTNode *)pho_alloc(sizeof(ASTNode), "ASTNode");
    if (!node) return NULL;
    
    node->type = type;
    node->value = NULL;
    if (value) {
        size_t len = strlen(value);
        char *val_copy = (char *)pho_alloc(len + 1, "ASTNode.value");
        if (val_copy) {
            memcpy(val_copy, value, len + 1);
            node->value = val_copy;
        }
    }
    node->num_value = 0.0;
    node->children = NULL;
    node->next = NULL;
    node->line = line;
    
    return node;
}

// Free an entire AST tree recursively
void ast_free(ASTNode *node) {
    if (!node) return;
    
    if (node->value) {
        pho_free((void *)node->value, "ASTNode.value");
    }
    
    ast_free(node->children);
    ast_free(node->next);
    
    pho_free(node, "ASTNode");
}

static const char *ast_type_to_string(ASTNodeType type) {
    switch (type) {
        case AST_PROGRAM: return "PROGRAM";
        case AST_LAYER_DECL: return "LAYER_DECL";
        case AST_FORWARD_BLOCK: return "FORWARD_BLOCK";
        case AST_TRAIN_BLOCK: return "TRAIN_BLOCK";
        case AST_INTERFERE_STMT: return "INTERFERE_STMT";
        case AST_ACTIVATE_STMT: return "ACTIVATE_STMT";
        case AST_EMIT_STMT: return "EMIT_STMT";
        case AST_IDENTIFIER: return "IDENTIFIER";
        case AST_NUMBER_LITERAL: return "NUMBER_LITERAL";
        case AST_STRING_LITERAL: return "STRING_LITERAL";
        case AST_PARAM_LIST: return "PARAM_LIST";
        case AST_CONFIG_PAIR: return "CONFIG_PAIR";
    }
    return "UNKNOWN";
}

// Pretty-print AST for debugging
void ast_print(const ASTNode *node, int indent) {
    if (!node) return;
    
    for (int i = 0; i < indent; i++) printf("  ");
    
    printf("[%s]", ast_type_to_string(node->type));
    if (node->value) {
        printf(" value: \"%s\"", node->value);
    }
    if (node->type == AST_NUMBER_LITERAL || node->num_value != 0.0) {
        printf(" num: %g", node->num_value);
    }
    printf("\n");
    
    ast_print(node->children, indent + 1);
    ast_print(node->next, indent); // Siblings are at the same indentation level
}
