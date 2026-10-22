# Test-only: links http_request into the test binaries (not the released llm extension).
# Used via: make release EXTRA_EXTENSION_CONFIGS=test/configs/http_request.cmake
duckdb_extension_load(http_request
    GIT_URL https://github.com/midwork-finds-jobs/duckdb_http_request
    GIT_TAG a6be835ef29c13957d017ac2e2afeb1cf9fb9616
)
duckdb_extension_statically_link(http_request)
