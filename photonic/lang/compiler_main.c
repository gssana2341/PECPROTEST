#include "ast.h"
#include "lexer.h"
#include "codegen.h"
#include <stdio.h>
#include <stdlib.h>

// Forward declaration of parser entrypoint
ASTNode *parser_parse(const char *source);

int main(int argc, char **argv) {
    printf("=== PhoLang Optical Compiler (phoc) ===\n\n");
    
    if (argc < 3) {
        printf("Usage: %s <input.pho> <output.c>\n", argv[0]);
        return 1;
    }
    
    const char *input_path = argv[1];
    const char *output_path = argv[2];
    
    printf("Reading source %s...\n", input_path);
    FILE *f = fopen(input_path, "r");
    if (!f) {
        fprintf(stderr, "Failed to open input file %s\n", input_path);
        return 1;
    }
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    if (len < 0) {
        fprintf(stderr, "Failed to determine file size for %s\n", input_path);
        fclose(f);
        return 1;
    }
    fseek(f, 0, SEEK_SET);
    
    char *source = malloc(len + 1);
    if (!source) {
        fprintf(stderr, "Memory allocation failed for source code\n");
        fclose(f);
        return 1;
    }
    size_t bytes_read = fread(source, 1, len, f);
    source[bytes_read] = '\0';
    fclose(f);
    
    printf("Parsing source code...\n");
    ASTNode *ast = parser_parse(source);
    if (!ast) {
        fprintf(stderr, "\n[Compiler Error] Parser failed!\n");
        free(source);
        return 1;
    }
    
    printf("Syntactic Analysis Completed. Printing AST:\n");
    ast_print(ast, 0);
    
    printf("\nGenerating highly optimized C target code into %s...\n", output_path);
    if (codegen_generate(ast, output_path) == 0) {
        printf("[CodeGen Success] C source code successfully written to %s!\n", output_path);
    } else {
        fprintf(stderr, "[CodeGen Failure] Transpilation process failed!\n");
        ast_free(ast);
        free(source);
        return 1;
    }
    
    ast_free(ast);
    free(source);
    printf("\nCompilation finished successfully. Memory cleaned (Zero Leaks).\n");
    return 0;
}
